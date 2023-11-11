# AE Decoder (ResNet-18 based) for Image Upsampling
# Uses the Resnet-18 Encoder Output after complete Downsampling (without(!) SPP)
# This Code is loosely based on a VAE-ResNet18 from Julian Stastny:
# https://github.com/julianstastny/VAE-ResNet18-PyTorch
import copy
import torch
import torch.nn.functional as F
from torch import nn
from ... util import _BNActConv, BasicBlockDec, ResizeConv2D, add, concat

# RGB mod
MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).permute(0, 3, 1, 2)#.to(device='cuda')
#MEAN = torch.tensor([[[[123, 117, 104]]]]).permute(0, 3, 1, 2)
STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).permute(0, 3, 1, 2)#.to(device='cuda')

class ResNetDecoder(nn.Module):

    def __init__(self, num_Blocks=[1, 1, 1, 1], z_dim=4, nc=3, decoders_skip_connections=None,
                 efficient=True, use_bn=True):
        super().__init__()
        self.use_bn=use_bn
        self.inplanes = 512
        self.efficient=efficient

        skip_connection_mode_l3 = None
        skip_connection_mode_l2 = None
        skip_connection_mode_l1 = None

        if decoders_skip_connections['mode'] == 'concatenate' or\
            decoders_skip_connections['mode'] == 'add':
            if decoders_skip_connections['layer_3']:
                skip_connection_mode_l3 = decoders_skip_connections['mode']
            # Layer 2
            if decoders_skip_connections['layer_2']:
                skip_connection_mode_l2 = decoders_skip_connections['mode']
            # Layer 1
            if decoders_skip_connections['layer_1']:
                skip_connection_mode_l1 = decoders_skip_connections['mode']
        else:
            pass

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2,
                                       skip_connection_mode_layer=None)  # No skip connections at this layer
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2,
                                       skip_connection_mode_layer=skip_connection_mode_l3)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2,
                                       skip_connection_mode_layer=skip_connection_mode_l2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1,
                                       skip_connection_mode_layer=skip_connection_mode_l1)

        self.conv1 = ResizeConv2D(64, nc, kernel_size=3, stride=2)

        if decoders_skip_connections != None:
            print('Deciders connected in AE_Decoder.py with parameters :', decoders_skip_connections)
        self.onexone_conv_l3 = _BNActConv(num_maps_in=512, num_maps_out=256, k=1)
        self.onexone_conv_l2 = _BNActConv(num_maps_in=512, num_maps_out=128, k=1)
        self.onexone_conv_l1 = _BNActConv(num_maps_in=512, num_maps_out=64, k=1)

    def _make_layer(self, block, planes, blocks, stride=1, skip_connection_mode_layer=None):
        layers = []

        # Create residual units.
        if blocks == 1:
            # Add the one and only residual unit. Pass the concatenation information. Pass the upsample module to it.
            layers += [block(self.inplanes, planes, stride, efficient=self.efficient, use_bn=self.use_bn,
                             mode=skip_connection_mode_layer)]
        elif blocks == 2:
            # Add the first residual unit. Pass the concatenation information.
            layers += [block(self.inplanes, self.inplanes, efficient=self.efficient, use_bn=self.use_bn,
                             mode=skip_connection_mode_layer)]
            # Add the last residual unit. Pass the upsample module to it.
            layers += [block(self.inplanes, planes, stride, efficient=self.efficient, use_bn=self.use_bn)]
        elif blocks > 2:
            # Add the first residual unit. Pass the concatenation information.
            layers += [block(self.inplanes, self.inplanes, efficient=self.efficient, use_bn=self.use_bn,
                             mode=skip_connection_mode_layer)]
            # Add additional residual units with standard configuration.
            for i in range(1, blocks - 1):
                layers += [block(self.inplanes, self.inplanes, efficient=self.efficient, use_bn=self.use_bn)]
            # Add the last residual unit. Pass the upsample module to it.
            layers += [block(self.inplanes, planes, stride, efficient=self.efficient, use_bn=self.use_bn)]

        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, features_segmentation_upsample=None, decoders_skip_connections=None):
        x = self.layer4(x)
        if features_segmentation_upsample == None:  # No connection between decoders
            x = self.layer3(x)
            x = self.layer2(x)
            x = self.layer1(x)
        elif decoders_skip_connections['mode'] == 'add' or\
                decoders_skip_connections['mode'] == 'concatenate':
            # Choose the skip operation
            if decoders_skip_connections['mode'] == 'add':
                skip_operation = add
            elif decoders_skip_connections['mode'] == 'concatenate':
                skip_operation = concat

            # Perform the skip operation if necessary
            if decoders_skip_connections['layer_3']:
                skip_segmentation_l3 = self.onexone_conv_l3(features_segmentation_upsample[0])
                skipped = skip_operation(x, skip_segmentation_l3)
                x = self.layer3(skipped)
            else:
                x = self.layer3(x)

            if decoders_skip_connections['layer_2']:
                skip_segmentation_l2 = self.onexone_conv_l2(features_segmentation_upsample[1])
                skipped = skip_operation(x, skip_segmentation_l2)
                x = self.layer2(skipped)
            else:
                x = self.layer2(x)

            if decoders_skip_connections['layer_1']:
                skip_segmentation_l1 = self.onexone_conv_l1(features_segmentation_upsample[2])
                skipped = skip_operation(x, skip_segmentation_l1)
                x = self.layer1(skipped)
            else:
                x = self.layer1(x)
        else:
            print('Invalid skip connection mode')

        # Interpolation after the four layers needs less resources
        x = F.interpolate(x, scale_factor=2)
        x = torch.sigmoid(self.conv1(x))

        # Zero Mean Normalisation to map the sigmoid Net Output 0-1 to -2.. - +2..
        # To Have the same Range as the Input Image
        x = x - MEAN
        x = x / STD

        return x
