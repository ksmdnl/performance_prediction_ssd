import copy
from itertools import chain

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url

from ....util import _Upsample, Bottleneck, BasicBlock, L2Norm
from ...ssd.detector_head_og import build_ssd_head

__all__ = ['ResNet', 'resnet18', 'resnet18_sws', 'resnet50_sws']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet18_sws': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth',
    'resnet50_sws': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth',
}

SUPPORTED_AE_DECODER = ['resnet10', 'resnet18', 'resnet34', 'espcn', 'basic', 'basic_noskip']

class ResNet(nn.Module):
    def __init__(self, AE_Decoder,
                 freeze_resnet,
                 block,
                 layers,
                 decoders_skip_connections,
                 modified_stride,
                 ssd_model,
                 num_classes,
                 *,
                 num_features=512, k_up=3,
                 efficient=True,
                 use_bn=True, spp_grids=(8, 4, 2, 1), spp_square_grid=False, **kwargs):
        super(ResNet, self).__init__()  # inherit from higher parents class
        self.inplanes = 64
        self.efficient = efficient
        self.AE_Decoder_Name = AE_Decoder
        self.freeze_ssd = freeze_resnet
        self.use_bn = use_bn
        self.modified_stride = modified_stride
        self.ssd_model = ssd_model
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # definition of the MaxPool2d function

        upsamples = []  # create an array for the different upsampling layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        

        self.decoders_skip_connections = decoders_skip_connections  # Dictionary of 'arguments' to connect decoders: NOT THE CONNECTION ARRAYS itselves
        if not self.freeze_ssd:
            self.detector_head == build_ssd_head(512, self.num_classes, False, False) # num classes for kitti to adjust

            if self.AE_Decoder_Name == 'resnet18':
                from .resnet import ResNetDecoder
                self.ae_decoder = ResNetDecoder(num_Blocks=[2, 2, 2, 2],
                                        decoders_skip_connections=self.decoders_skip_connections)
            elif self.AE_Decoder_Name == 'basic' or self.AE_Decoder_Name == 'basic_noskip':
                from .basic import BasicDecoder
                self.ae_decoder = BasicDecoder(use_skips=False if 'noskip' in self.AE_Decoder_Name else True)
            else:
                ValueError(f"The decoder type {self.AE_Decoder_Name} is not supported.")
            

        if self.ssd_model == "ssd_mod_sec":
            self.L2Norm = L2Norm(256,20)
            self.layer3_mod = copy.deepcopy(self.layer3)
        elif self.ssd_model == "ssd":
            self.L2Norm = L2Norm(128,20)
        if not self.freeze_ssd:
            self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.detector_head]
        else:
            self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        self.spp_size = num_features

        self.upsample = nn.ModuleList(list(reversed(upsamples)))

        self.random_init = [self.upsample]
            
        if not self.freeze_ssd:
            self.random_init += self.ae_decoder

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # TODO?
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:  # if you want BatchNormalization in your layer
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def mod_strides(self, layers):
        """
            The function acts like a switch that changes the stride of second layer
            in the third res block during the multitask learning.
        """
        conv5_block1 = layers[0]
        conv5_block1.conv1.stride = (1, 1)
        conv5_block1.conv2.stride = (1, 1)
        conv5_block1.downsample[0].stride = (1, 1)
        return layers

    def reset(self):
        conv5_block1 = self.layer4[0]
        conv5_block1.conv1.stride = (2, 2)
        conv5_block1.conv2.stride = (1, 1)
        conv5_block1.downsample[0].stride = (2, 2)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    # Forward Path Residual-Block to build one Block from two ResUnits
    def forward_resblock(self, x, layers, mod):
        skip = None
        if mod:
            layers = self.mod_strides(layers)
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x  # the activated output is x and the output before activation is skip
        return x, skip

    # Forward-Down Path with two different outputs (single task training)
    # First define the internal layers for each normal ResUnit: conv1 --> bn1 --> act1 --> maxpool
    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # Build the full Resnet-18, use the forward_resblock function for constructing a block witch includes two ResUnits
        # features (list) contains the lateral outputs from the 3 Blocks and the SPP output (build from the 4th block)
        # features_normal (Tensor) includes only the downsampling-output from the last (fourth) block
        # The skip outputs correspond to the non-activated output of a residual unit
        # 'source' contains the first and second feature maps for object detection
        features = []
        source = list()
        mod = False
        x, skip = self.forward_resblock(x, self.layer1, mod)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2, mod)
        x_source = x
        if self.ssd_model == "ssd":
            x_source = self.L2Norm(x_source)
            source.append(x_source)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3, mod)
        mod = False if not self.ssd_model == "ssd_mod_sec" else True
        # if self.modified_stride:
        if self.ssd_model == "ssd_mod_sec":
            x_source, _ = self.forward_resblock(x_source, self.layer3_mod, mod)
            x_source = self.L2Norm(x_source)
            source.append(x_source)
        else:
            source.append(x)
        mod = False
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4, mod=False)
        # if self.ssd_model = "ssd_mod_bn":
        #     x, skip = self.forward_resblock(x, self.layer4, mod)
        # if self.modified_stride:
        if self.ssd_model == "ssd_mod_sec":
            x_source, _ = self.forward_resblock(x_source, self.layer4, mod)
            source.append(x_source)
        elif self.ssd_model == "ssd_mod_bn":
            mod = True
            x_source, _ = self.forward_resblock(source[0], self.layer4, mod)
            source.append(x_source)
            self.reset()
        elif self.ssd_model == "ssd":
            source.append(x)
        features_normal = x  # x = output of last residual unit (not activated), skip = activated output of last residual unit
        features += [skip]#[self.spp.forward(skip)]

        return features, features_normal, source

    # Forward Path for standard upsampling for the semantic segmentation
    # Uses only the connections contained in the feature-list from forward_down

    def forward_up(self, features):
        features = features[::-1]
        x = features[0]
        # Def. Upsampling
        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    # Upsamling with the new AE-Decoder
    def forward_ae_decoder(self, features, image_size,
                           features_segmentation_upsample=None,  # Not None Only for ResNet AE Decoder
                           decoders_skip_connections=None,  # Not None Only for ResNet AE Decoder
                           ):
        if not self.freeze_ssd:
            if self.AE_Decoder_Name == 'basic' or self.AE_Decoder_Name == 'basic_noskip':
                return self.ae_decoder(features, image_size)

            elif 'resnet' in self.AE_Decoder_Name:
                return self.ae_decoder(features,
                                       features_segmentation_upsample,  # Feature maps for skip connections
                                       decoders_skip_connections,  # Feature maps argumetns for skip connections
                                       )
            else:
                return self.ae_decoder(features)
        else:
            return 0

    # Master forward function with Feature-Output (logits) for the Sem. Seg. and the upsampled Image from the AE-Decoder
    # Returning Argument is a Python-Dict contains both outputs together
    def forward(self, image):
        if self.freeze_ssd:  # AE-Decoder is in the SemSeg file
            # only_skips as alternative, AE doesnt need spp activation
            #only_skips, resnet_output = self.forward_down(image)
            only_skips, resnet_output, source = self.forward_down(image)
            #print(resnet_output.size())
            # features_without_spp contains features of RB1, RB2, RB3, and SPP (NOTE: No RB4 features here)
            upsampled_features = self.forward_up(only_skips)#(skips_and_spp)  # Output :  x, {'features': features, 'upsamples': upsamples}
            # features here is just input to forward_up function
            # i.e. features_with_spp

            dict_AutoSwiftNet = {'upsampled_features': upsampled_features,#'logits_for_SemSeg': logits,
                                 'features_without_spp': resnet_output,
                                 'features_with_spp': only_skips,
                                 'decoders_skip_connections_args': self.decoders_skip_connections,
                                 'detection_source': source
                                 # NOTE: 'decoders_skip_connections_args' contains only arguments dictionary and
                                 # not the actual feature maps for skip connections between two decoders.
                                 # Feature maps for skip connections between two decoders are in logits[1]['upsamples']
                                 }
            return dict_AutoSwiftNet

        else:
            only_skips, resnet_output, source = self.forward_down(image)
            detection_output = self.detector_head(source)
            # features_without_spp contains features of RB1, RB2, RB3, and SPP (NOTE: No RB4 features here)
            upsampled_features = self.forward_up(only_skips)  # Output :  x, {'features': features, 'upsamples': upsamples}
            # features here is just input to forward_up function
            # i.e. features_with_spp

            forward_output_dictionary = upsampled_features[1]  # forward_output_dictionary = {'features': features, 'upsamples': upsamples}
            features_segmentation_upsample = forward_output_dictionary['upsamples']

            if self.AE_Decoder_Name == 'basic' or self.AE_Decoder_Name == 'basic_noskip':
                ae_decoder_output = self.forward_ae_decoder(skips_and_spp, image.shape[2:4])
            elif 'resnet' in self.AE_Decoder_Name:
                ae_decoder_output = self.forward_ae_decoder(resnet_output,
                                                            image.shape[2:4],
                                                            features_segmentation_upsample,
                                                            # Feature maps for skip connections
                                                            self.decoders_skip_connections,
                                                            # Argument dictoinary for skip connections
                                                            )
                # print('Decoders connected')
            else:
                ae_decoder_output = self.forward_ae_decoder(resnet_output, image.shape[2:4])

            dict_AutoSwiftNet = {'detection_output': detection_output,
                                 'autoencoder_output': ae_decoder_output,
                                 'upsampled_features': upsampled_features}
            return dict_AutoSwiftNet


# Semi-Supervised and Semi-Weakly Supervised (SWS) ImageNet ResNet-18 Models
# ResNet-18 and ResNet-34 use the BasisBlock, however ResNet-50 or higher a Bottleneck-Block
# **kwargs allow to pass keyworded! variable length! of arguments to a function

def resnet18(AE_Decoder, freeze_resnet, decoders_skip_connections, pretrained=True, **kwargs):
    '''
    Constructs a ResNet-18 model with symmetric block representation
    Args: pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNet(AE_Decoder, freeze_resnet, BasicBlock, [2, 2, 2, 2], decoders_skip_connections, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet18_sws(AE_Decoder, freeze_resnet, decoders_skip_connections, pretrained=True, **kwargs):
    '''
    Constructs a ResNet-18 SWS model with symmetric block representation (Semi-Weakly Supervised)
    Args: pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNet(AE_Decoder, freeze_resnet, BasicBlock, [2, 2, 2, 2], decoders_skip_connections, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18_sws'], progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(AE_Decoder, freeze_resnet, decoders_skip_connections, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(AE_Decoder, freeze_resnet, Bottleneck, [3, 4, 6, 3], decoders_skip_connections, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet50_sws(AE_Decoder, freeze_resnet, decoders_skip_connections, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(AE_Decoder, freeze_resnet, Bottleneck, [3, 4, 6, 3], decoders_skip_connections, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_sws'], progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model

'''Dummy Main for Testing'''
if __name__ == "__main__":
    print("Main Starting")
    AutoSwiftNet_Test = resnet18_sws()