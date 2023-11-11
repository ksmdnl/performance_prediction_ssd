import torch
import torch.nn as nn
from . ODRec import SSDRec
from . resnet.full_network_single_scale import resnet18_sws, resnet18

# --------------------------------------------------------------------------------
#  Custom function for the specific model architecture to load/update state_dict
# --------------------------------------------------------------------------------
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
    def __init__(self, num_classes_wo_bg, AE_Decoder, freeze_resnet, decoders_skip_connections, modified_stride, ssd_model, do_efficient):

        super().__init__()
        use_bn = True

        # Create the SwiftNet based AutoSwiftNet-Model
        encoder = resnet18(AE_Decoder=AE_Decoder,
                      decoders_skip_connections=decoders_skip_connections,
                      freeze_resnet=freeze_resnet,
                      modified_stride=modified_stride,
                      ssd_model=ssd_model,
                      num_classes=num_classes_wo_bg,
                      efficient=do_efficient)
        # Skip connections in SemSegModel directly via backbone : AutoSwiftNet_model
        # Freeze Options for the complete backbone (ResNet-Encoder) --> Look at the options im SemsegModel
        self.loaded_model = SSDRec(AE_Decoder, encoder, num_classes_wo_bg,
                    decoders_skip_connections=decoders_skip_connections,
                    use_bn=use_bn,
                    freeze_complete_backbone=freeze_resnet,
                    freeze_backbone_bn=freeze_resnet,
                    freeze_backbone_bn_affine=freeze_resnet,
                    modified_stride=modified_stride,
                    ssd_model=ssd_model,
                    efficient=do_efficient)
        # bug?
        #self.loaded_model.eval()

    def forward(self, batch):
        ###batch = batch.to('cuda') # Just for FLOPs Evaluation

        dict_AutoSwiftNet_output = self.loaded_model.forward(batch)
        return dict_AutoSwiftNet_output

    def random_init_params(self):
        return self.loaded_model.random_init_params()

    def fine_tune_params(self):
        return self.loaded_model.fine_tune_params()

    def freeze_parameters(self):
        return self.loaded_model.freeze_parameters()

if __name__ == '__main__':

    """    skip_connections_decoders_dict = {'layer_1': 0,
                                       'layer_2': 0,
                                       'layer_3': 0,
                                       'mode': 'add',
                                       }
    init_all = Creation(options=opt, args=args)
    model = AutoSwiftNet(19, 'resnet18', True, decoders_skip_connections=skip_connections_decoders_dict)"""
    #print(model)