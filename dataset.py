from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from pathlib import Path
import torch


class FemurDataset(Dataset):
    def __init__(self, dataframe, image_size=256, augment=False):
        self.dataframe = dataframe
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        xray_path = row["xrays"]
        xray_image = Image.open(xray_path).convert("L")

        mask_path = row["masks"]
        mask_image = Image.open(mask_path).convert("L")

        ### resize x-rays and masks
        xray_image = TF.resize(
            xray_image,
            (self.image_size, self.image_size)
        )

        mask_image = TF.resize(
            mask_image,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST
        )

        if self.augment and torch.rand(1).item() < 0.5:
            xray_image = TF.hflip(xray_image)
            mask_image = TF.hflip(mask_image)

        xray_tensor = TF.to_tensor(xray_image)
        mask_tensor = TF.to_tensor(mask_image)

        ### make mask binary
        mask_tensor = (mask_tensor > 0.5).float()

        return xray_tensor, mask_tensor


class FemurTestDataset(Dataset):
    def __init__(self, dataframe, image_size=256):
        self.dataframe = dataframe
        self.image_size = image_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        xray_path = row["xrays"]
        xray_image = Image.open(xray_path).convert("L")

        ### resize test x-rays
        xray_image = TF.resize(
            xray_image,
            (self.image_size, self.image_size)
        )

        filename = Path(xray_path).name

        xray_tensor = TF.to_tensor(xray_image)

        return xray_tensor, filename