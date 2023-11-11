import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
import sys
sys.path.append('../')
#from resnet import resnet


mbboxes = {
	'300': [4, 6, 6, 6, 4, 4],
	'512': [4, 6, 6, 6, 4, 4, 4],
	'275': [4, 6, 6, 6, 4, 4, 4],
}


class L2Norm(nn.Module):

	def __init__(self, n_channels, scale):
		super(L2Norm, self).__init__()
		self.n_channels = n_channels
		self.gamma = scale or None
		self.eps = 1e-10
		self.weight = nn.Parameter(torch.Tensor(self.n_channels))
		self.reset_parameters()

	def reset_parameters(self):
		init.constant_(self.weight, self.gamma)
		

	def forward(self, x):
		norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
		x = torch.div(x, norm)
		x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
		
		return x

def Extras(size):	
	layers = []

	if size == 300:
		conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
		conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
		conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1) 
		conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
		conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
		conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
		conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
		conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)

		layers = [conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2, conv11_1, conv11_2]
		return layers
	else:
		conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
		conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
		conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1) 
		conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
		conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
		conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
		conv11_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
		conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
		conv12_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
		if size != 512:
			# kitti ssd upscaled
			conv12_2 = nn.Conv2d(128, 256, kernel_size=(2,4), stride=1)#, padding=1)	
		else:
			# for ssd512
			conv12_2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)	

		layers = [conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2, conv11_1, conv11_2, conv12_1, conv12_2]
		
		return layers

def PredictionLayer(extra_layers, bboxes, num_classes, head_bn):
	loc_layers = []
	conf_layers = []
	if not head_bn:
		loc_layers += [nn.Conv2d(128, bboxes[0] * 4, kernel_size=3, padding=1)]	
		loc_layers += [nn.Conv2d(256, bboxes[1] * 4, kernel_size=3, padding=1)]
		conf_layers += [nn.Conv2d(128, bboxes[0] * num_classes, kernel_size=3, padding=1)]
		conf_layers += [nn.Conv2d(256, bboxes[1] * num_classes, kernel_size=3, padding=1)]
	else:
		loc_layers += [
						nn.Sequential(nn.Conv2d(256, bboxes[0] * 4, kernel_size=3, padding=1),
						nn.BatchNorm2d(bboxes[0] * 4))
					   ]	
		loc_layers += [
						nn.Sequential(nn.Conv2d(512, bboxes[1] * 4, kernel_size=3, padding=1),
						nn.BatchNorm2d(bboxes[1] * 4))
					  ]
		conf_layers += [
						nn.Sequential(nn.Conv2d(256, bboxes[0] * num_classes, kernel_size=3, padding=1),
					    nn.BatchNorm2d(bboxes[0] * num_classes))]
		conf_layers += [
						nn.Sequential(nn.Conv2d(512, bboxes[1] * num_classes, kernel_size=3, padding=1),
						nn.BatchNorm2d(bboxes[1] * num_classes))
						]
	
	for k, v in enumerate(extra_layers[1::2], 2):
		if not head_bn:
			loc_layers += [nn.Conv2d(v.out_channels, bboxes[k] * 4, kernel_size=3, padding=1)]
			conf_layers += [nn.Conv2d(v.out_channels, bboxes[k] * num_classes, kernel_size=3, padding=1)]
		else:
			loc_layers += [nn.Sequential(nn.Conv2d(v.out_channels, bboxes[k] * 4, kernel_size=3, padding=1),
							nn.BatchNorm2d(bboxes[k] * 4))]
			conf_layers += [nn.Sequential(nn.Conv2d(v.out_channels, bboxes[k] * num_classes, kernel_size=3, padding=1),
							nn.BatchNorm2d(bboxes[k] * num_classes))]
			
	return loc_layers, conf_layers

class SSD_Head(nn.Module):
	def __init__(self, num_classes, bboxes, ssd_size, pretrain=None, head_bn=True):
		super(SSD_Head, self).__init__()
	
		self.num_classes = num_classes
		self.bboxes = bboxes
		self.size = ssd_size
		self.head_bn = head_bn	
		self.extra_list = Extras(self.size)
		self.pred_layers = PredictionLayer(self.extra_list, self.bboxes, self.num_classes, self.head_bn)
		self.loc_layers_list, self.conf_layers_list = self.pred_layers
		self.corpus = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
			nn.ReLU(inplace=True),
			nn.Conv2d(1024, 1024, kernel_size=1),
			nn.ReLU(inplace=True)
		)

		self.extras = nn.ModuleList(self.extra_list)
		self.loc = nn.ModuleList(self.loc_layers_list)
		self.conf = nn.ModuleList(self.conf_layers_list)

	def forward(self, source):
		
		loc_pred = []
		conf_pred = []
		mylist = []
		x = source[-1]
		source = source[:-1]
		x = self.corpus(x)
		for i, v in enumerate(self.extras):
			x = F.relu(v(x), inplace=True)
			if i % 2 == 1: 							# detection layers
				source.append(x)
		for s, l, c in zip(source, self.loc, self.conf):
			loc_pred.append(l(s).permute(0, 2, 3, 1).contiguous())		
			conf_pred.append(c(s).permute(0, 2, 3,1).contiguous())

		loc_pred = torch.cat([o.view(o.size(0), -1) for o in loc_pred], 1)
		conf_pred = torch.cat([o.view(o.size(0), -1) for o in conf_pred], 1)

		# bounding box coordinate
		loc_pred = loc_pred.view(loc_pred.size(0), -1, 4)

		# confidence score for each class
		conf_pred = conf_pred.view(conf_pred.size(0), -1, self.num_classes)

		return loc_pred, conf_pred

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

