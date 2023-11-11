import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
import sys
sys.path.append('../')

mbboxes = {
	'300': [4, 6, 6, 6, 4, 4],
	'512': [4, 6, 6, 6, 4, 4, 4],
	'275': [4, 6, 6, 6, 4, 4, 4],
}
	
output_channels = [1024, 512, 512, 256, 256, 128]
output_predictions = [256, 512, 512, 256, 256, 128]

class SSD_Head(nn.Module):
	def __init__(self, num_classes, bboxes, ssd_size, pretrain=None, head_bn=True):
		super(SSD_Head, self).__init__()
	
		self.num_classes = num_classes
		self.bboxes = bboxes
		self.size = ssd_size
		self.head_bn = head_bn			
		self.corpus = nn.Sequential(
			nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), # input N, 512, 32, 32
			nn.ReLU(inplace=True),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=6, dilation=6),
			nn.ReLU(inplace=True)
		)
		self._build_additional_features(output_channels)

		self.loc = []
		self.conf = []

		for nd, oc in zip(self.bboxes, output_predictions):
			self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
			self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

		self.loc = nn.ModuleList(self.loc)
		self.conf = nn.ModuleList(self.conf)

	def forward(self, source):
		
		loc_pred = []
		conf_pred = []
		last_res_map = source[-1]
		source = source[:-1]
		last_res_map = self.corpus(last_res_map)
		for layer in self.additional_blocks:
			last_res_map = layer(last_res_map)
			source.append(last_res_map)

		for s, l, c in zip(source, self.loc, self.conf):
			loc_pred.append(l(s).permute(0, 2, 3, 1).contiguous())		
			conf_pred.append(c(s).permute(0, 2, 3,1).contiguous())

		loc_pred = torch.cat([o.view(o.size(0), -1) for o in loc_pred], 1)
		conf_pred = torch.cat([o.view(o.size(0), -1) for o in conf_pred], 1)

		# bounding box coordinate, and confidence score for each class
		loc_pred = loc_pred.view(loc_pred.size(0), -1, 4)
		conf_pred = conf_pred.view(conf_pred.size(0), -1, self.num_classes)

		return loc_pred, conf_pred

	def _build_additional_features(self, input_size):
			self.additional_blocks = []
			for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
				if i < 3:
					layer = nn.Sequential(
						nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
						nn.BatchNorm2d(channels),
						nn.ReLU(inplace=True),
						nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
						nn.BatchNorm2d(output_size),
						nn.ReLU(inplace=True),
					)
				else:
					layer = nn.Sequential(
						nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
						nn.BatchNorm2d(channels),
						nn.ReLU(inplace=True),
						nn.Conv2d(channels, output_size, kernel_size=2, stride=2, bias=False),
						nn.BatchNorm2d(output_size),
						nn.ReLU(inplace=True),
					)

				self.additional_blocks.append(layer)

			self.additional_blocks = nn.ModuleList(self.additional_blocks)

def build_ssd_head(size=300, num_classes=21, pretrain=False, head_bn=False):
	ssd_size = size
	if size not in [300,512,275]:
		print("ERROR: You specified size " + repr(size) + ". However, " +
			 "currently only SSD300 and SSD512 is supported!")
	return SSD_Head(num_classes=num_classes,
			   bboxes=mbboxes[str(size)],
			   ssd_size=ssd_size,
			   pretrain=pretrain,
			   head_bn=head_bn)