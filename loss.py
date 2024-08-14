import torch.nn as nn
from torchvision.models import vgg19
import config


class VGGLoss(nn.Module):
    """
    VGGLoss uses a pretrained VGG19 network to compute feature-based loss.

    This loss function is used in super-resolution models to enhance the visual
    quality of the generated images by comparing the features extracted from
    the generated image and the target high-resolution image.
    """

    def __init__(self):
        """
        Initializes the VGGLoss module.

        Loads a pretrained VGG19 network and extracts the first 35 layers for
        feature extraction. The VGG19 network is set to evaluation mode and
        its parameters are frozen to prevent updates during training.
        """
        super().__init__()

        try:
            self.vgg = vgg19(pretrained=True).features[:35].eval().to(config.DEVICE)
        except Exception as e:
            print(f"Error loading VGG19 model: {e}")
            raise e

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

    def forward(self, input, target):
        """
        Computes the forward pass of the VGGLoss.

        Args:
            input (torch.Tensor): The generated image tensor.
            target (torch.Tensor): The target high-resolution image tensor.

        Returns:
            torch.Tensor: The computed MSE loss between the features of the input and target images.
        """
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)
