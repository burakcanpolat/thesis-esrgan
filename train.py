import os
import torch
import config
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from ESRGAN import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


def train_fn(
        loader,
        disc,
        gen,
        opt_gen,
        opt_disc,
        l1,
        vgg_loss,
        g_scaler,
        d_scaler,
        writer,
        tb_step,
):
    """
    Function to train the GAN model for one epoch.

    Args:
        loader (DataLoader): DataLoader for the dataset.
        disc (nn.Module): The discriminator model.
        gen (nn.Module): The generator model.
        opt_gen (Optimizer): Optimizer for the generator.
        opt_disc (Optimizer): Optimizer for the discriminator.
        l1 (nn.L1Loss): L1 loss function.
        vgg_loss (VGGLoss): VGG loss function.
        g_scaler (torch.cuda.amp.GradScaler): Gradient scaler for generator (for mixed precision training).
        d_scaler (torch.cuda.amp.GradScaler): Gradient scaler for discriminator (for mixed precision training).
        writer (SummaryWriter): TensorBoard writer for logging.
        tb_step (int): TensorBoard step counter.

    Returns:
        int: Updated TensorBoard step counter.
    """
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        with torch.cuda.amp.autocast('cpu'):
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        loss_critic.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake))
            loss_for_vgg = vgg_loss(fake, high_res)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_disc.step()

        writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
        tb_step += 1

        if idx % 100 == 0 and idx > 0:
            plot_examples("test_images/", gen)

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss.item(),
            vgg=loss_for_vgg.item(),
            adversarial=adversarial_loss.item(),
        )

    return tb_step


def validate_fn(
        loader,
        disc,
        gen,
        l1,
        vgg_loss,
):
    """
    Function to validate the GAN model for one epoch.

    Args:
        loader (DataLoader): DataLoader for the validation dataset.
        disc (nn.Module): The discriminator model.
        gen (nn.Module): The generator model.
        l1 (nn.L1Loss): L1 loss function.
        vgg_loss (VGGLoss): VGG loss function.

    Returns:
        tuple: (avg_l1_loss, avg_vgg_loss, avg_val_loss) - Average losses over the validation dataset.
    """
    loop = tqdm(loader, leave=True)
    gen.eval()
    disc.eval()
    total_l1_loss = 0
    total_vgg_loss = 0
    total_val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for idx, (low_res, high_res) in enumerate(loop):
            high_res = high_res.to(config.DEVICE)
            low_res = low_res.to(config.DEVICE)

            fake = gen(low_res)
            l1_loss = l1(fake, high_res)
            loss_for_vgg = vgg_loss(fake, high_res)
            val_loss = l1_loss + loss_for_vgg

            total_l1_loss += l1_loss.item()
            total_vgg_loss += loss_for_vgg.item()
            total_val_loss += val_loss.item()
            num_batches += 1

            loop.set_postfix(
                l1=l1_loss.item(),
                vgg=loss_for_vgg.item(),
                val_loss=val_loss.item(),
            )

    avg_l1_loss = total_l1_loss / num_batches
    avg_vgg_loss = total_vgg_loss / num_batches
    avg_val_loss = total_val_loss / num_batches

    gen.train()
    disc.train()

    return avg_l1_loss, avg_vgg_loss, avg_val_loss


def main():
    """
    Main function to set up the training process, including data loading, model initialization,
    and starting the training loop.
    """
    train_dataset = MyImageFolder(
        hr_dir="data/DIV2K_HR/train",  # Yüksek çözünürlüklü eğitim görüntülerinin bulunduğu dizin
        lr_dir="data/DIV2K_LR/train",  # Düşük çözünürlüklü eğitim görüntülerinin bulunduğu dizin
        transform=config.highres_transform,  # Yüksek çözünürlüklü görüntüler için dönüşümler
        lowres_transform=config.lowres_transform,  # Düşük çözünürlüklü görüntüler için dönüşümler
        both_transforms=config.both_transforms,  # Hem yüksek hem düşük çözünürlüklü görüntüler için dönüşümler
    )
    val_dataset = MyImageFolder(
        hr_dir="data/DIV2K_HR/val",  # Yüksek çözünürlüklü doğrulama görüntülerinin bulunduğu dizin
        lr_dir="data/DIV2K_LR/val",  # Düşük çözünürlüklü doğrulama görüntülerinin bulunduğu dizin
        transform=config.highres_transform,  # Yüksek çözünürlüklü görüntüler için dönüşümler
        lowres_transform=config.lowres_transform,  # Düşük çözünürlüklü görüntüler için dönüşümler
        both_transforms=None,  # Doğrulama seti için ek dönüşümler yok
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )

    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    writer = SummaryWriter("logs")
    tb_step = 0
    l1 = nn.L1Loss()
    gen.train()
    disc.train()
    vgg_loss = VGGLoss()

    g_scaler = torch.cuda.amp.GradScaler('cpu')
    d_scaler = torch.cuda.amp.GradScaler('cpu')

    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if config.LOAD_MODEL:
        load_checkpoint(
            os.path.join(checkpoint_dir, config.CHECKPOINT_GEN),
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(checkpoint_dir, config.CHECKPOINT_DISC),
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        tb_step = train_fn(
            train_loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
            vgg_loss,
            g_scaler,
            d_scaler,
            writer,
            tb_step,
        )

        avg_l1_loss, avg_vgg_loss, avg_val_loss = validate_fn(
            val_loader,
            disc,
            gen,
            l1,
            vgg_loss,
        )

        writer.add_scalar("Validation L1 Loss", avg_l1_loss, global_step=epoch)
        writer.add_scalar("Validation VGG Loss", avg_vgg_loss, global_step=epoch)
        writer.add_scalar("Validation Total Loss", avg_val_loss, global_step=epoch)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=os.path.join(checkpoint_dir, "gen.pth"))
            save_checkpoint(disc, opt_disc, filename=os.path.join(checkpoint_dir, "disc.pth"))


if __name__ == "__main__":
    try_model = False

    if try_model:
        # Will just use pretrained weights and run on images
        # in test_images/ and save the ones to SR in saved/
        gen = Generator(in_channels=3).to(config.DEVICE)
        opt_gen = optim.Adam(
            gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9)
        )
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        plot_examples("test_images/", gen)
    else:
        # This will train from scratch
        main()
