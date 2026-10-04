import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_PATH = "img.png"     # image to convolve 


def make_sobel_x(size=3):

    smooth = np.array([math.comb(size - 1, i) for i in range(size)], dtype=np.float32)
    deriv = np.convolve([math.comb(size - 2, i) for i in range(size - 1)], [-1, 1])
    return torch.from_numpy(np.outer(smooth, deriv).astype(np.float32))


SOBEL_X = make_sobel_x(3)


def apply_kernel(image, kernel, stride=1, padding=0):
  
    weight = kernel.view(1, 1, *kernel.shape)      # conv2d - (out_ch, in_ch, kh, kw)
    return F.conv2d(image, weight, stride=stride, padding=padding)





def task_3_1():
    print(" Sobel convolution")

    # Loading the image as one grayscale channel 
    gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Could not read {IMAGE_PATH}")
    img = torch.from_numpy(gray).float().div(255.0).unsqueeze(0).unsqueeze(0)
    print("Input shape:      ", tuple(img.shape))

    # padding=1 with a 3x3 kernel keeps the output the same size as the input
    feature_map = apply_kernel(img, SOBEL_X, stride=1, padding=1)
    print("Feature map shape:", tuple(feature_map.shape))

    # Saving the feature map as an image 
    arr = feature_map[0, 0].abs().numpy()
    cv2.imwrite("feature_map.png", (arr / arr.max() * 255).astype(np.uint8))
    print("Saved feature_map.png")



class FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 input channels (RGB) -> 16 output channels, 3x3 kernel, stride 1, padding 0
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        # 2x2 window moving in steps of 2 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        print("Input:       ", tuple(x.shape))

        # Conv: 16 filters, each spans all 3 input channels, so channels go 3 -> 16.
        # With a 3x3 kernel, stride 1 and no padding, each spatial axis loses
        # kernel - 1 = 2 pixels: (416 + 2*0 - 3) / 1 + 1 = 414.
        x = self.conv(x)
        print("After Conv:  ", tuple(x.shape))

        # ReLU: sets negative values to 0. It doesn't change the
        # shape, so channels (16) and spatial size (414 x 414) stay the same.
        x = self.relu(x)
        print("After ReLU:  ", tuple(x.shape))

        # MaxPool: keeps the largest value in each 2x2 window and moves by 2, so height and
        # width are halved: floor((414 - 2) / 2) + 1 = 207. Channels stay 16 because pooling
        # is applied to each channel separately.
        x = self.pool(x)
        print("After MaxPool:", tuple(x.shape))
        return x


def task_3_2():
    print("\nTask 3.2")
    block = FeatureExtraction()
    dummy = torch.randn(1, 3, 416, 416)       # (batch, RGB channels, height, width)
    block(dummy)


def main():
    task_3_1()
    task_3_2()


if __name__ == "__main__":
    main()
