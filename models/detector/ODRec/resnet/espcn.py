# AE Decoder (ESPCN based) for Image Upsampling
# ESPCN developed from Wenzhe Shi et. al. (Twitter)
# "Real-Time Single Image and Video Super-Resolution Using an EfficientSub-Pixel Convolutional Neural Network"
# https://arxiv.org/pdf/1609.05158.pdf
# ESPCNet based on SISR = Single Image Super Resolution
# This Code based in parts on the PyTorch Implementation of the ESPCN from Liu Changyu:
# https://github.com/Lornatang/ESPCN-PyTorch

import torch
import torch.nn as nn

MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).permute(0, 3, 1, 2).to(device='cuda')
STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).permute(0, 3, 1, 2).to(device='cuda')

# Second Shot
class ESPCN(nn.Module):
    def __init__(self, scale_factor=4):
        super(ESPCN, self).__init__()

        # Die beste Lösung aus Johannes Arbeit:
        # Ansatz 4 --> FM + SP(2) + FM + SP(4) + SP(4)
        # Feature maps 1
        self.feature_maps_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Sub-pixel convolution layer 1
        self.sub_pixel_1 = nn.Sequential(
            nn.Conv2d(256, 128 * (2 ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh()
        )

        # Feature maps 2
        self.feature_maps_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh()

        )

        # Sub-pixel convolution layer 2
        self.sub_pixel_2 = nn.Sequential(
            nn.Conv2d(64, 32 * (4 ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            nn.Tanh()
        )

        # Sub-pixel convolution layer 3
        self.sub_pixel_3 = nn.Sequential(
            nn.Conv2d(32, 3 * (4 ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            nn.Sigmoid()
        )

        # ToDo: Weiterer Ansatz der zu testen wäre
        # Ansatz NEU --> FM + SP(2) + FM + SP(2) + FM + SP(2) + FM + SP(8)
        # Feature maps 1
        # self.feature_maps_1 = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh()
        # )
        #
        # # Sub-pixel convolution layer 1
        # self.sub_pixel_1 = nn.Sequential(
        #     nn.Conv2d(256, 128 * (2 ** 2), kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2),
        #     nn.Tanh()
        # )
        #
        # # Feature maps 2
        # self.feature_maps_2 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh()
        #
        # )
        #
        # # Sub-pixel convolution layer 2
        # self.sub_pixel_2 = nn.Sequential(
        #     nn.Conv2d(64, 32 * (2 ** 2), kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2),
        #     nn.Tanh()
        # )
        #
        # # Feature maps 3
        # self.feature_maps_3 = nn.Sequential(
        #     nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh()
        #
        # )
        #
        # # Sub-pixel convolution layer 3
        # self.sub_pixel_3 = nn.Sequential(
        #     nn.Conv2d(16, 3 * (8 ** 2), kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(8),
        #     nn.Tanh()
        # )



    def forward(self, inputs):

        inputs = self.feature_maps_1(inputs)

        inputs = self.sub_pixel_1(inputs)

        inputs = self.feature_maps_2(inputs)

        inputs = self.sub_pixel_2(inputs)

        # inputs = self.feature_maps_3(inputs)

        inputs = self.sub_pixel_3(inputs)
        output = inputs

        output = output - MEAN
        output = output / STD
        return output

