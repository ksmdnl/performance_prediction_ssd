# This ist the Standard SwiftNet Decoder modified for Image Upsampling as an Autoencoder
# It also called Basis-Decoder with lateral skip connections

import torch
import torch.nn as nn
from itertools import chain
from ...util import _UpsampleBlend, _Upsample, upsample, _BNActConv, BasicBlock, Bottleneck

MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).permute(0, 3, 1, 2).to(device='cuda')
STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).permute(0, 3, 1, 2).to(device='cuda')

# Class Basic-Decoder for Image Upsampling
class BasicDecoder(nn.Module):
    def __init__(self, block=BasicBlock, *, num_features=128, k_up=3, use_bn=True, use_skips=True, **kwargs):

        super().__init__()  # inherit from higher parents class
        self.inplanes = 64
        self.use_bn = use_bn  # use Batch Normalization
        upsamples = []  # create an array for the different upsampling layers
        self.logits = _BNActConv(num_maps_in=128, num_maps_out=3, batch_norm=use_bn)

        # self._make_layer(block, X) adjusts self.inplanes accordingly
        if use_skips:
            upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
            self._make_layer(block, 128)
            upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
            self._make_layer(block, 256)
            upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
            self._make_layer(block, 512)
        else:
            upsamples += [_UpsampleBlend(num_features, use_bn=self.use_bn)]
            self._make_layer(block, 128)
            upsamples += [_UpsampleBlend(num_features, use_bn=self.use_bn)]
            self._make_layer(block, 256)
            upsamples += [_UpsampleBlend(num_features, use_bn=self.use_bn)]
            self._make_layer(block, 512)

        self.upsample = nn.ModuleList(list(reversed(upsamples)))
        self.random_init = self.upsample
        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # TODO?
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes):
        self.inplanes = planes * block.expansion

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    # Forward Path for Image Upsampling to BGR Format
    def forward(self, features, image_size):
        features = features[::-1] # reverse the list of features
        x = features[0] # take the first element after the reverse operation
        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]

        SemSegInput = x, {'features': features, 'upsamples': upsamples}

        feats, additional = zip(*[SemSegInput])
        logits = self.logits.forward(feats[0])
        upsampled = upsample(self.training, logits, image_size)

        # Sigmoid-Function for a limited Output Range between 0 and 1
        upsampled = torch.sigmoid(upsampled)

        output = upsampled - MEAN
        output = output / STD

        return output
