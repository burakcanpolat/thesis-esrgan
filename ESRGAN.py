import torch
from torch import nn
from torch.nn import Conv2d


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of a convolutional layer followed by an optional activation function.

    Attributes:
        cnn (nn.Conv2d): Convolutional layer.
        act (nn.Module): Activation function (LeakyReLU or Identity).
    """

    def __init__(self, in_channels, out_channels, use_activation, **kwargs):
        """
        Initializes the ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_activation (bool): Whether to use an activation function.
            **kwargs: Additional arguments for the Conv2d layer.
        """
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution and activation.
        """
        return self.act(self.cnn(x))


class UpsampleBlock(nn.Module):
    """
    An upsample block that increases the resolution of the input tensor.

    Attributes:
        upsample (nn.Upsample): Upsampling layer.
        conv (nn.Conv2d): Convolutional layer.
        act (nn.LeakyReLU): Activation function.
    """

    def __init__(self, in_channels, scale_factor=2):
        """
        Initializes the UpsampleBlock.

        Args:
            in_channels (int): Number of input channels.
            scale_factor (int): Factor by which to scale the input resolution.
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = Conv2d(in_channels, in_channels, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Forward pass of the UpsampleBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Upsampled and convolved tensor with activation.
        """
        return self.act(self.conv(self.upsample(x)))


class DenseResidualBlock(nn.Module):
    """
    A dense residual block consisting of multiple ConvBlocks with dense connections.

    Attributes:
        residual_beta (float): Scaling factor for the residual connection.
        blocks (nn.ModuleList): List of ConvBlocks.
    """

    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        """
        Initializes the DenseResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of channels for each ConvBlock.
            residual_beta (float): Scaling factor for the residual connection.
        """
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + i * channels,
                    channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_activation=True if i <= 3 else False,
                )
            )

    def forward(self, x):
        """
        Forward pass of the DenseResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with dense residual connections.
        """
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)

        return self.residual_beta * out + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB) consisting of multiple DenseResidualBlocks.

    Attributes:
        residual_beta (float): Scaling factor for the residual connection.
        rrdb (nn.Sequential): Sequential container of DenseResidualBlocks.
    """

    def __init__(self, in_channels, residual_beta=0.2):
        """
        Initializes the RRDB.

        Args:
            in_channels (int): Number of input channels.
            residual_beta (float): Scaling factor for the residual connection.
        """
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        """
        Forward pass of the RRDB.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with residual connections.
        """
        return self.residual_beta * self.rrdb(x) + x


class Generator(nn.Module):
    """
    Generator network for super-resolution.

    Attributes:
        initial (nn.Conv2d): Initial convolutional layer.
        residuals (nn.Sequential): Sequential container of RRDBs.
        conv (nn.Conv2d): Convolutional layer after RRDBs.
        upsamples (nn.Sequential): Sequential container of UpsampleBlocks.
        final (nn.Sequential): Final convolutional layers with activation.
    """

    def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
        """
        Initializes the Generator.

        Args:
            in_channels (int): Number of input channels.
            num_channels (int): Number of channels for intermediate layers.
            num_blocks (int): Number of RRDB blocks.
        """
        super().__init__()
        self.initial = nn.Conv2d(
            in_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels), UpsampleBlock(num_channels),
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        """
        Forward pass of the Generator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Super-resolved output tensor.
        """
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsamples(x)
        return self.final(x)


class Discriminator(nn.Module):
    """
    Discriminator network to distinguish between real and generated images.

    Attributes:
        blocks (nn.Sequential): Sequential container of ConvBlocks.
        classifier (nn.Sequential): Sequential container for classification layers.
    """

    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        """
        Initializes the Discriminator.

        Args:
            in_channels (int): Number of input channels.
            features (list of int): List of output channels for each ConvBlock.
        """
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    use_activation=True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        """
        Forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Discrimination score for the input tensor.
        """
        x = self.blocks(x)
        return self.classifier(x)


def initialize_weights(model, scale=0.1):
    """
    Initializes weights of the model using Kaiming normal initialization.

    Args:
        model (nn.Module): The model to initialize.
        scale (float): Scaling factor for the weights.
    """

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()


def test():
    """
    Tests the Generator and Discriminator networks.

    Creates a batch of random low-resolution images, passes them through the
    Generator to produce high-resolution images, and then passes the generated
    images through the Discriminator to get the discrimination scores.

    """
    gen = Generator()
    disc = Discriminator()
    low_res = 24
    x = torch.randn((5, 3, low_res, low_res))
    gen_out = gen(x)
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)


if __name__ == "__main__":
    test()
    print("Test passed!")
