import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class MyImageFolder(Dataset):
    """
    Custom Dataset class for loading high-resolution (HR) and low-resolution (LR) images from directories,
    applying transformations, and returning both versions of the images.

    Attributes:
        hr_dir (str): Directory containing the high-resolution images.
        lr_dir (str): Directory containing the low-resolution images.
        transform (callable, optional): Transformation function for high-resolution images.
        lowres_transform (callable, optional): Transformation function for low-resolution images.
        both_transforms (callable, optional): Transformation function to be applied to both versions.
        images (list): List of image file names in the directories.
    """

    def __init__(
        self,
        hr_dir,
        lr_dir,
        transform=None,
        lowres_transform=None,
        both_transforms=None,
    ):
        """
        Initializes the MyImageFolder dataset.

        Args:
            hr_dir (str): Directory containing the high-resolution images.
            lr_dir (str): Directory containing the low-resolution images.
            transform (callable, optional): Transformation function for high-resolution images.
            lowres_transform (callable, optional): Transformation function for low-resolution images.
            both_transforms (callable, optional): Transformation function to be applied to both versions.
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.lowres_transform = lowres_transform
        self.both_transforms = both_transforms
        self.images = os.listdir(
            hr_dir
        )  # Assumes HR and LR directories have matching file names

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves a high-resolution image and its corresponding low-resolution version,
        applying the specified transformations.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (low_res, high_res) where low_res is the low-resolution
                   version of the image and high_res is the high-resolution version.
        """
        img_name = self.images[idx]
        hr_img_path = os.path.join(self.hr_dir, img_name)
        lr_img_path = os.path.join(
            self.lr_dir, img_name.replace(".png", "x4m.png")
        )  # Assuming the LR images have 'x4m' suffix

        hr_image = Image.open(hr_img_path).convert("RGB")
        lr_image = Image.open(lr_img_path).convert("RGB")

        if self.both_transforms:
            try:
                hr_image = self.both_transforms(image=np.array(hr_image))["image"]
                lr_image = self.both_transforms(image=np.array(lr_image))["image"]
            except Exception as e:
                print(f"Error applying both_transforms: {e}")

        if self.transform:
            try:
                hr_image = self.transform(image=np.array(hr_image))["image"]
            except Exception as e:
                print(f"Error applying transform: {e}")
        if self.lowres_transform:
            try:
                lr_image = self.lowres_transform(image=np.array(lr_image))["image"]
            except Exception as e:
                print(f"Error applying lowres_transform: {e}")

        return lr_image, hr_image