from __future__ import annotations

import time
import matplotlib.pyplot
import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class convReLU(nn.Module):
    """Main Down block within the UNET Model"""

    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv1 = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_out)
        self.conv2 = nn.Conv2d(num_out, num_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU()

    def forward(self, input_data) -> torch.Tensor:
        out = self.conv1(input_data)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UpBlock(nn.Module):

    def __init__(self, num_in, num_out):
        """Main Up block within the UNET MODEL"""
        super().__init__()
        self.up = nn.ConvTranspose2d(num_in, num_out, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.block = convReLU(num_out + num_out, num_out)

    def forward(self, x, skip):
        out = self.up(x)
        out = torch.cat([out, skip], axis=1)

        return self.block.forward(out)


class MainNetwork(nn.Module):
    """UNET Model that follows the well documented UNET design. Can be refractored to reduce line count but as
    this was an intro level model it was kept to specifically show how the layerws worked"""

    def __init__(self):
        super().__init__()
        # encodes
        self.block1 = convReLU(1, 64)
        self.block2 = convReLU(64, 128)
        self.block3 = convReLU(128, 256)
        self.block4 = convReLU(256, 512)

        self.skip_block = convReLU(512, 1024)

        # decodes
        self.block5 = UpBlock(1024, 512)
        self.block6 = UpBlock(512, 256)
        self.block7 = UpBlock(256, 128)
        self.block8 = UpBlock(128, 64)

        self.pool = nn.MaxPool2d((2, 2))

        #make tensor 4
        self.final_conv = nn.Conv2d(64, 4, 1)

    def forward(self, x):
        """ Main forward block routing each block to the next and copying skip blocks and applying as needed"""

        out = self.block1.forward(x)
        copy_crop_one = out.clone()
        out = self.pool(out)

        out = self.block2.forward(out)
        copy_crop_two = out.clone()
        out = self.pool(out)

        out = self.block3.forward(out)
        copy_crop_three = out.clone()
        out = self.pool(out)

        out = self.block4.forward(out)
        copy_crop_four = out.clone()
        out = self.pool(out)

        out = self.skip_block.forward(out)

        out = self.block5.forward(out, copy_crop_four)

        out = self.block6.forward(out, copy_crop_three)

        out = self.block7.forward(out, copy_crop_two)

        out = self.block8.forward(out, copy_crop_one)

        out = self.final_conv.forward(out)

        return out



class ImageLoader(Dataset):
    """
    Loads and transforms the raw images into a usable image set
    """
    def __init__(self, main_dir_slice, main_dir_seg):
        self.main_dir_slice = main_dir_slice
        self.total_imgs_slice = os.listdir(main_dir_slice)

        self.main_dir_seg = main_dir_seg
        self.total_imgs_seg = os.listdir(main_dir_seg)
        self.total_imgs_slice.sort()
        self.total_imgs_seg.sort()

    def __len__(self):
        return len(self.total_imgs_slice)

    def __getitem__(self, idx):

        # convert to tensor
        trans = transforms.ToTensor()

        # load in as grey scale
        img_slice = os.path.join(self.main_dir_slice, self.total_imgs_slice[idx])
        slice = trans(Image.open(img_slice).convert("L"))

        img_seg = os.path.join(self.main_dir_seg, self.total_imgs_seg[idx])
        seg = trans(Image.open(img_seg).convert("L"))

        # One Hot Encoding for Segmentation Mapping
        t = transforms.ConvertImageDtype(torch.int8)
        seg = t(seg).div(42, rounding_mode = 'trunc').type(torch.int8)

        return slice, seg


# from https://github.com/pytorch/pytorch/issues/1249
def dice_loss(input, target):
    """
    calculate Dice loss between two images, not used
    """
    smooth = 0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))









