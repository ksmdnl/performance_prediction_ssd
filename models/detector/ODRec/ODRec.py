import copy
import torch.nn as nn
from itertools import chain

from ...util import upsample, _BNActConv, L2Norm

from .resnet.basic import BasicDecoder 
from ..ssd.detector_head_og import build_ssd_head
from ...rec_decoders.swiftnet_rec.rec_decoders.swiftnet import SwiftNetDecoder
from ... util import upsample, _BNActConv

class SSDRec(nn.Module):
	def __init__(self, rec_decoder,
		backbone,
		num_classes,
		decoders_skip_connections,
		freeze_complete_backbone=False,
		freeze_backbone_bn=False,
		freeze_backbone_bn_affine=False,
		modified_stride=False,
		ssd_model="ssd",
		efficient=True,
	):
		super(SSDRec, self).__init__()
		self.rec_decoder_name = rec_decoder
		self.backbone = backbone
		self.target_size = (2048, 1024)  # todo: use
		# self.image_size = img_size
		self.num_classes = num_classes
		self.freeze_ssd = freeze_complete_backbone
		self.freeze_backbone_bn = freeze_backbone_bn
		self.freeze_backbone_bn_affine = freeze_backbone_bn_affine
		self.modified_stride = modified_stride
		self.ssd_model = ssd_model
		self.efficient = efficient

		print('\nFreezing the Semantic Segmentation:       ', freeze_complete_backbone)
		print('Decoder-Architecture for the Autoencoder: ', self.rec_decoder_name)

		self.decoders_skip_connections = decoders_skip_connections  # Dictionary of 'arguments' to connect decoders: NOT THE CONNECTION ARRAYS itselves

		if self.freeze_ssd:
			# Get SSD detector head
			self.detector_head = build_ssd_head(512, self.num_classes, False, False)
			# self.rec_decoder = BasicDecoder(use_skips=False if 'noskip' in self.rec_decoder_name else True)
			self.rec_decoder = SwiftNetDecoder(
				use_skips=False if 'noskip' in self.rec_decoder_name else True,
				use_spp=False if 'nospp' in self.rec_decoder_name else True,
				inplanes=self.backbone.dims
			)
		
	def train(self, mode=True):
		"""
		Override the default train() to freeze the BN and/or entire backbone parameters
		"""
		super(SSDRec, self).train(mode)

		if self.freeze_ssd:
			for i, child in enumerate(self.backbone.children()):
				if i < 9:
					child.eval()
					for param in child.parameters():
						param.requires_grad = False
			self.detector_head.eval()

		if self.freeze_ssd:
			for i, child in enumerate(self.backbone.children()):
				if i < 8:
					for m in child.modules():
						if isinstance(m, nn.BatchNorm2d):
							m.eval()
							if self.freeze_backbone_bn_affine:
								m.weight.requires_grad = False
								m.bias.requires_grad = False

	def forward(self, batch):
		backbone_out = self.backbone(batch)
		detection_output = self.detector_head(backbone_out["source"])

		upsamples = backbone_out['semseg']['upsamples']
		features = [backbone_out['spp_input'], backbone_out['skips_and_spp']]
		reconstructed_im = self.rec_decoder(features, batch.shape[2:4], upsamples)

		outputs = {'detection_output': detection_output, 'reconstructed_im': reconstructed_im}

		return outputs

	def random_init_params(self):
		if self.freeze_ssd:
			return chain(*([self.rec_decoder.parameters(), self.backbone.random_init_params()]))
		else:
			return chain(*([self.backbone.random_init_params()]))

	def fine_tune_params(self):
		if self.freeze_ssd:
			return chain(*([self.backbone.fine_tune_params(), self.detector_head.parameters()]))
		else:
			return self.backbone.fine_tune_params()
