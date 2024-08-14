import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants for configuration
LOAD_MODEL = False  # Flag to determine whether to load the model from a checkpoint
SAVE_MODEL = True  # Flag to determine whether to save the model checkpoints
CHECKPOINT_GEN = "gen.pth"  # File path to save/load generator model checkpoint
CHECKPOINT_DISC = "disc.pth"  # File path to save/load discriminator model checkpoint
DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Determine the device to use (GPU if available, else CPU)
LEARNING_RATE = 1e-4  # Learning rate for training
NUM_EPOCHS = 10000  # Number of epochs for training
BATCH_SIZE = 16  # Batch size for training
LAMBDA_GP = 10  # Lambda parameter for gradient penalty
NUM_WORKERS = 4  # Number of workers for data loading
HIGH_RES = 128  # High resolution size
LOW_RES = HIGH_RES // 4  # Low resolution size derived from high resolution
IMG_CHANNELS = 3  # Number of image channels (e.g., 3 for RGB)

# Transformations for high resolution images
highres_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize image
        ToTensorV2(),  # Convert image to tensor
    ]
)

# Transformations for low resolution images
lowres_transform = A.Compose(
    [
        A.Resize(
            width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC
        ),  # Resize image to low resolution
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize image
        ToTensorV2(),  # Convert image to tensor
    ]
)

# Common transformations for both high and low resolution images
both_transforms = A.Compose(
    [
        A.RandomCrop(
            width=HIGH_RES, height=HIGH_RES
        ),  # Randomly crop image to high resolution size
        A.HorizontalFlip(
            p=0.5
        ),  # Randomly flip image horizontally with 50% probability
        A.RandomRotate90(
            p=0.5
        ),  # Randomly rotate image by 90 degrees with 50% probability
    ]
)

# Transformations for test images
test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # Normalize image
        ToTensorV2(),  # Convert image to tensor
    ]
)
