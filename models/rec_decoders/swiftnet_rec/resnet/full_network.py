import torch.utils.model_zoo as model_zoo

from .... encoders.resnet.full_network import ResNet
from .... util import BasicBlock, Bottleneck

__all__ = ['ResNetRec']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

# Definition of the Class "ResNet": base for the "SwiftNet"
class ResNetRec(ResNet):
    def __init__(self, block, layers, *,
        task="segmentation",
        num_features=128,
        k_up=3,
        efficient=True,
        use_bn=True,
        spp_grids=(8, 4, 2, 1),
        spp_square_grid=False,
        lateral=False,
        **kwargs
    ):
        super().__init__(block, layers, task=task, num_features=num_features, k_up=k_up, efficient=efficient, use_bn=use_bn,
                         spp_grids=spp_grids, spp_square_grid=spp_square_grid)  # inherit from higher parents class

    # Overwrite forward_down function
    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # Build the full backbone
        features = []
        source = []
        x, lat = self.forward_resblock(x, self.layer1)
        features += [lat]
        x, lat = self.forward_resblock(x, self.layer2)
        features += [lat]
        if self.task == "detection":
            source.append(lat) # torch.Size([1, 128, 64, 64])
        x, lat = self.forward_resblock(x, self.layer3)
        features += [lat]
        if self.task == "detection":
            source.append(lat) # torch.Size([1, 256, 32, 32])
        x, lat = self.forward_resblock(x, self.layer4)
        if self.task == "detection":
            source.append(lat) # torch.Size([1, 512, 16, 16])
        # x : activated output of RB4, lat : non-activated output of RB4 used for lateral connections

        # One would assume that spp works on the activated output, however, this is not the case.
        # This also align with Orsic figure 3 (https://arxiv.org/abs/1903.08469). Here the vertical/lateral connections
        # (so the ones going from top to down) correspond to lat and the horizontal connections correspond to x.
        features += [self.spp.forward(lat)]
        return features, x, lat, source

    # Master forward function with feature output (logits) for segmentation and the upsampled image reconstruction
    # Returning Argument is a Python-Dict contains both outputs together
    def forward(self, image):
        skips_and_spp, backbone_output, spp_input, source = self.forward_down(image)

        semseg_prelogits = self.forward_up(skips_and_spp)
        semseg_dict = {'prelogits': semseg_prelogits[0],
                       'features': semseg_prelogits[1]['features'],
                       'upsamples': semseg_prelogits[1]['upsamples']}

        # Note that this dictionary only contains SwiftNet-specific outputs for now
        dict_ = {'semseg': semseg_dict,
                 'backbone_output': backbone_output,
                 'spp_input': spp_input,
                 'skips_and_spp': skips_and_spp,
                 'source': source}

        return dict_

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetRec(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetRec(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model