import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SmokeDataset(Dataset):
    def __init__(self, images, masks=None, img_transforms=None, mask_transforms=None):
        self.images = images
        self.masks = masks
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.image_files = []
        self.mask_files = []

        if self.masks is not None:
            assert len(os.listdir(self.images)) == len(os.listdir(self.masks))
            self.image_files = sorted(os.listdir(self.images))
            self.mask_files = sorted(os.listdir(self.masks))
        else:
            self.image_files = sorted(os.listdir(self.images))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = os.path.join(self.images, self.image_files[idx])
        img = Image.open(image_name)
        trans = transforms.ToTensor()

        if self.img_transforms is not None:
            img = self.img_transforms(img)
        else:
            img = trans(img)

        if self.masks is not None:
            mask_name = os.path.join(self.masks, self.mask_files[idx])
            mask = Image.open(mask_name)
            if self.mask_transforms is not None:
                mask = self.mask_transforms(mask)
            else:
                mask = trans(mask)
            mask_max = mask.max().item()
            mask /= mask_max
            return img, mask
        return img