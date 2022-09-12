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
        super().__init__()
        self.up = nn.ConvTranspose2d(num_in, num_out, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.block = convReLU(num_out + num_out, num_out)

    def forward(self, x, skip):
        out = self.up(x)
        out = torch.cat([out, skip], axis=1)

        return self.block.forward(out)


class MainNetwork(nn.Module):

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
        trans = transforms.ToTensor()

        img_slice = os.path.join(self.main_dir_slice, self.total_imgs_slice[idx])
        slice = trans(Image.open(img_slice).convert("L"))

        img_seg = os.path.join(self.main_dir_seg, self.total_imgs_seg[idx])
        seg = trans(Image.open(img_seg).convert("L"))
        
        t = transforms.ConvertImageDtype(torch.int8)
        seg = t(seg).div(42, rounding_mode = 'trunc').type(torch.int8)

        return slice, seg


def train_model():
    train_slices_loader = ImageLoader("keras_png_slices_data/keras_png_slices_train",
                                      "keras_png_slices_data/keras_png_slices_seg_train")
    train_slices = DataLoader(train_slices_loader, batch_size=4, shuffle=True, num_workers= 4)

    validate_slices_loader = ImageLoader("keras_png_slices_data/keras_png_slices_validate",
                                         "keras_png_slices_data/keras_png_slices_seg_validate")
    validate_slices = DataLoader(validate_slices_loader, batch_size=4, shuffle=True, num_workers= 4)

    model = MainNetwork()
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(30):
        start_time = time.time()

        running_loss = 0
        model.train(True)
        for index, data in enumerate(tqdm(train_slices)):
            x, y = data
            x = x.to(dtype = torch.float32)
            y = y.to(dtype = torch.long)
            y = y.cuda()
            x = x.cuda()
            x.requires_grad = True

            optimizer.zero_grad()
            outputs = model(x).cuda()

            loss = criterion(outputs, y.squeeze(1)).cuda()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(loss.item())

            if index % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        running_loss = 0
        count = 0
        model.train(False)
        with torch.no_grad():
            for index, data in enumerate(validate_slices):
                x, y = data
                x = x.to(dtype = torch.float32)
                y = y.to(dtype = torch.long)
                y = y.cuda()
                x = x.cuda()
                outputs = model(x)
                loss = criterion(outputs, y.squeeze(1)).cuda()
                running_loss += loss
                count += 1

        print("Validation Loss is:", running_loss / count)
        print("Finished Epoch in: ", time.time() - start_time)

    torch.save(model, "Attempt11")


# from https://github.com/pytorch/pytorch/issues/1249
def dice_loss(input, target):
    smooth = 0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def calculate_dice_loss():
    model = torch.load('Attempt10')
    model.eval()

    test_slices_loader = ImageLoader("keras_png_slices_data/keras_png_slices_test",
                                     "keras_png_slices_data/keras_png_slices_seg_test")
    test_slices = DataLoader(test_slices_loader, batch_size=1, shuffle=True)

    count = 0
    running_dice = 0
    with torch.no_grad():
        for index, data in enumerate(test_slices):
            x, y = data
            x, y = x.to(0, dtype = torch.float32), y.to(0, dtype = torch.long)
            outputs = model(x)
            outputs = outputs.squeeze()
            outputs = torch.argmax(outputs, 0)

            second_total = 0
            for dimension in range(4):
                
                out = (outputs == dimension).bool().int()
                y1 = (y == dimension).bool().int()
                loss = dice_loss(out, y1).item()
                running_dice += loss
                second_total += loss
                count += 1
            if second_total / 4 > 0.1:
                print("FAILED")
                
    print(1 - running_dice / count)               

def show_comparision():
    model = torch.load('Attempt10')
    model.eval()

    test_slices_loader = ImageLoader("keras_png_slices_data/keras_png_slices_test",
                                     "keras_png_slices_data/keras_png_slices_seg_test")
    test_slices = DataLoader(test_slices_loader, batch_size=1, shuffle=True)
    fig = plt.figure(figsize=(10, 7))
    
    for index, data in enumerate(test_slices):
        x,y = data    
        
        output_image = model.forward(x.cuda())
        output_image = output_image[0]

        output_image  = torch.argmax(output_image, 0)

        fig.add_subplot(1, 3, 1)
        plt.imshow(x[0].permute(1,2,0).detach().cpu().numpy())
        plt.title("Test Image Non Segmented (X)")
        fig.add_subplot(1, 3, 2)
        plt.imshow(y[0].permute(1,2,0).detach().cpu().numpy())
        plt.title("Reference Test Image Segmented (Y)")
        fig.add_subplot(1, 3, 3)
        plt.imshow(output_image.detach().cpu().numpy())
        plt.title("Generated Segementation Map")
        break
    plt.show()


if __name__ == '__main__':
    train_model()
    #calculate_dice_loss()
    #show_comparision()





