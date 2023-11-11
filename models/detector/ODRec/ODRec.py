import copy
import torch.nn as nn
from itertools import chain

from ..util import upsample, _BNActConv, L2Norm
#from ..ssd.detector_head import build_ssd_head

class SSDRec(nn.Module):
	def __init__(self, AE_Decoder,
				 backbone,
				 num_classes,
				 decoders_skip_connections,
				 freeze_complete_backbone=False,
				 freeze_backbone_bn=False,
				 freeze_backbone_bn_affine=False,
				 use_bn=True,
				 modified_stride=False,
				 ssd_model="ssd",
				 efficient=True):
		super(SSDRec, self).__init__()
		self.AE_Decoder_Name = AE_Decoder
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
		print('Decoder-Architecture for the Autoencoder: ', self.AE_Decoder_Name)

		self.decoders_skip_connections = decoders_skip_connections  # Dictionary of 'arguments' to connect decoders: NOT THE CONNECTION ARRAYS itselves

		if self.freeze_ssd:
			# Get SSD detector head
			if self.ssd_model == "ssd_mod_sec":
				from ..ssd.detector_head import build_ssd_head
				self.detector_head = build_ssd_head(512, self.num_classes, False, False) # num classes for kitti to adjust
			elif self.ssd_model == "ssd":
				from ..ssd.detector_head_og import build_ssd_head
				self.detector_head = build_ssd_head(512, self.num_classes, False, False) # num classes for kitti to adjust
			# elif self.ssd_model == "ssd_mod":
			# 	ValueError("not supported")
			elif self.ssd_model == "ssd_mod_bn":
				from ..ssd.detector_head_bn import build_ssd_head
				self.detector_head = build_ssd_head(512, self.num_classes, False, False) # num classes for kitti to adjust
			if self.AE_Decoder_Name == 'resnet10':
				from .resnet.resnet import ResNetDecoder  # New ae-decoder-object from class ResNet18Decoder
				self.ae_decoder = ResNetDecoder(num_Blocks=[1, 1, 1, 1],
												decoders_skip_connections=self.decoders_skip_connections,
												efficient=self.efficient)
			elif self.AE_Decoder_Name == 'resnet18':
				from .resnet.resnet import ResNetDecoder  # New ae-decoder-object from class ResNet18Decoder
				self.ae_decoder = ResNetDecoder(num_Blocks=[2, 2, 2, 2],
												decoders_skip_connections=self.decoders_skip_connections,
												efficient=self.efficient)
			elif self.AE_Decoder_Name == 'resnet34':
				from .resnet.resnet import ResNetDecoder  # New ae-decoder-object from class ResNet18Decoder
				self.ae_decoder = ResNetDecoder(num_Blocks=[3, 4, 6, 3],
												decoders_skip_connections=self.decoders_skip_connections,
												efficient=self.efficient)
			elif self.AE_Decoder_Name == 'espcn':
				from .resnet.espcn import ESPCN  # New ae-decoder-object from class ESPCN in folder AE Decoder ESPCN
				self.ae_decoder = ESPCN()
			elif self.AE_Decoder_Name == 'basic' or self.AE_Decoder_Name == 'basic_noskip':
				from .resnet.basic import BasicDecoder  # New ae-decoder-object from class Basis_Dec (SwiftNet)
				self.ae_decoder = BasicDecoder(use_skips=False if 'noskip' in self.AE_Decoder_Name else True)
			else:  # Basis Decoder
				ValueError(f"The decoder type {self.AE_Decoder_Name} is not supported.")

		# use the new universal BNActConv for different Activation-Functions
		#self.logits = _BNActConv(backbone.num_features, num_classes, batch_norm=use_bn) # redundant

	# Upsamling with the new AE-Decoder
	def forward_ae_decoder(self, features, image_size,
						   lateral_features=None,
						   decoders_skip_connections_args=None,
						   ):
		# if self.freeze_ssd:
		if 'basic' in self.AE_Decoder_Name:
			return self.ae_decoder(features, image_size)
		elif 'resnet' in self.AE_Decoder_Name:
			return self.ae_decoder(features, lateral_features, decoders_skip_connections_args)
		else:
			ValueError(f"The decoder type {self.AE_Decoder_Name} is not supported.")

	def train(self, mode=True):
		"""
		Override the default train() to freeze the BN and/or entire backbone parameters
		"""
		super(SSDRec, self).train(mode)

		if self.freeze_ssd:
			# print("Freezing ResNet backbone via eval() and param.requires_grad = False.")
			for i, child in enumerate(self.backbone.children()):
				if i < 9:
					child.eval()
					for param in child.parameters():
						param.requires_grad = False
			# self.backbone.eval()
			self.detector_head.eval()

		if self.freeze_ssd:
			# print("Freezing Mean/Var of BatchNorm2D in backbone.")
			for i, child in enumerate(self.backbone.children()):
				if i < 8:
					for m in child.modules():
						if isinstance(m, nn.BatchNorm2d):
							m.eval()
							if self.freeze_backbone_bn_affine:
								# print("Freezing Weight/Bias of BatchNorm2D backbone.")
								m.weight.requires_grad = False
								m.bias.requires_grad = False

	def forward(self, batch):
		AutoSwiftNet_dict = self.backbone(batch)
		upsampled_feature = AutoSwiftNet_dict[
			'upsampled_features']  # SemSegInput: x, {'features': features, 'upsamples': upsamples}
		# for f in upsampled_feature[1]:
		# 	print(f.size())
		#--------------------------------- Redundant for ODRec --------------------------------------#
		# Compute Mask for the Semantic Segmentation
		#feats, additional = zip(*[
		#	SemSegInput])  # results in tuples: feats = (x,) and additional = ({'features': features, 'upsamples': upsamples}, )
		#logits = self.logits.forward(feats[0])  # feats = (x,) hence feats[0] = x
		#upsampled = upsample(self.training, logits, batch.shape[2:4])
		#--------------------------------------------------------------------------------------------#
		if self.freeze_ssd:
			# Get the first and second features for detection
			source = AutoSwiftNet_dict['detection_source']
			detection_output = self.detector_head(source)
			#--------------------------------- Redundant for ODRec ---------------------------------#
			# Getting the features for the skip connections between the two decoders
			segmentation_forward_output_dictionary = upsampled_feature[1]  # {'features': features, 'upsamples': upsamples}
			lateral_features = segmentation_forward_output_dictionary['upsamples']
			#----------------------------------------------------------------------------------------#
			decoders_skip_connections_args = AutoSwiftNet_dict['decoders_skip_connections_args']
			if 'basic' in self.AE_Decoder_Name:
				features_with_spp = AutoSwiftNet_dict['features_with_spp']
				ae_decoder_output = self.forward_ae_decoder(features_with_spp, batch.shape[2:4])
			elif self.AE_Decoder_Name == 'espcn':
				features_without_spp = AutoSwiftNet_dict['features_without_spp']
				ae_decoder_output = self.forward_ae_decoder(features_without_spp, batch.shape[2:4])

			elif 'resnet' in self.AE_Decoder_Name:
				features_without_spp = AutoSwiftNet_dict['features_without_spp']
				# TODO: Add if else conditions for skip connections between two decoders (CHECK THIS!)
				# ae_decoder_output = self.forward_ae_decoder(features_without_spp, batch.shape[2:4])
				ae_decoder_output = self.forward_ae_decoder(features_without_spp,
															batch.shape[2:4],
															lateral_features,
															# Feature maps for skip connections between decoders
															decoders_skip_connections_args,  # Skip connection arguments
															)
			else:
				ValueError(f"The decoder type {self.AE_Decoder_Name} is not supported.")

			# The Following command is just necessary for evaluation the Feature Maps of the Encoder at the end
			# features_without_spp = AutoSwiftNet_dict['features_without_spp']
			# dict_AutoSwiftNet_output = {'output_SemSeg': upsampled, 'output_autoencoder': ae_decoder_output,
			#                             'features_without_spp': features_without_spp}
			dict_AutoSwiftNet_output = {'detection_output': detection_output,
										'output_autoencoder': ae_decoder_output}

		else:
			ae_decoder_output = AutoSwiftNet_dict['autoencoder_output']
			detection_output = AutoSwiftNet_dict['detection_output']
			dict_AutoSwiftNet_output = {'detection_output': detection_output,
										'output_autoencoder': ae_decoder_output}

		return dict_AutoSwiftNet_output

	def random_init_params(self):
		if self.freeze_ssd:
			return chain(*([self.ae_decoder.parameters(), self.backbone.random_init_params()]))
		else:
			return chain(*([self.backbone.random_init_params()]))  # if ae_decoder is not used

	def fine_tune_params(self):
		if self.freeze_ssd:
			return chain(*([self.backbone.fine_tune_params(), self.detector_head.parameters()]))
		else:
			return self.backbone.fine_tune_params()
