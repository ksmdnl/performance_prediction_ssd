import torch.nn as nn
from . ODRec import SSDRec
from ... encoders.resnet.full_network import resnet18

def load_state_dict_into_model(model, pretrained_dict):
    #ToDo: This function may throw errors, if you not use the correct value for Freeze_SemSeg
    model_dict = model.state_dict()
    if list(pretrained_dict.keys())[0] == 'backbone.conv1.weight':
        pretrained_dict = {k.replace('backbone', 'loaded_model.backbone'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('logits', 'loaded_model.logits'): v for k, v in pretrained_dict.items()}
    for name, param in pretrained_dict.items():
        if name not in model_dict:
            print("State_dict mismatch!", flush=True)
            continue
        model_dict[name].copy_(param)
    model.load_state_dict(pretrained_dict, strict=False)

# Class "AutoSwiftNet" based on SwiftNet and an separate Autoencoder-Decoder for Image-Upsampling
class ODReconstruction(nn.Module):
    def __init__(self, num_classes_wo_bg,
        rec_decoder,
        freeze_resnet,
        decoders_skip_connections,
        modified_stride,
        ssd_model,
        do_efficient,
    ):

        super().__init__()
        use_bn = True

        self.encoder = resnet18(pretrained=True, efficient=False, use_bn=use_bn)
        self.model = SSDRec(rec_decoder, self.encoder, num_classes_wo_bg,
                    decoders_skip_connections=decoders_skip_connections,
                    freeze_complete_backbone=freeze_resnet,
                    freeze_backbone_bn=freeze_resnet,
                    freeze_backbone_bn_affine=freeze_resnet,
                    modified_stride=modified_stride,
                    ssd_model=ssd_model,
        )
        self.model.eval()

    def forward(self, batch):
        outputs = self.model.forward(batch)
        return outputs

    def random_init_params(self):
        return self.model.random_init_params()

    def fine_tune_params(self):
        return self.model.fine_tune_params()

    def freeze_parameters(self):
        return self.model.freeze_parameters()