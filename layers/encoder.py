# """
# 	reference :
# 				https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# 				https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# """

# import torch.nn as nn
# from layers.drn import Bottleneck, BasicBlock, DRN
# import torch.utils.model_zoo as model_zoo
# import math
# import torch



# e_config = {

# 	'VGG16':{
# 			'layers' : [['64', '64', 'M'], ['128','128', 'M'], ['256', '256', '256', 'M'],
# 				['512', '512', '512', 'M'], ['512' , '512', '512', 'M']],
# 			'scale_factor' : 32,
# 			'type' : 'VGG',


# 		},
# 	'DVGG16' :{
# 			'layers' : [['64', '64', 'M'], ['128','128', 'M'], ['256', '256', '256', 'M'],
# 				['512', '512', '512'], ['512d', '512d', '512d']],
# 			'scale_factor' : 8,
# 			'type' : 'VGG',
# 		},


# 	'DRN26': {
# 			'arch' : 'C',
# 			'layers' : [1, 1, 2, 2, 2, 2, 1, 1],
# 			'type' : 'DRN',
# 			'block' : BasicBlock,
# 			'scale_factor' : 8,

# 	},
# 	'DRN42': {
# 			'arch' : 'C',
# 			'layers' : [1, 1, 3, 4, 6, 3, 1, 1],
# 			'type' : 'DRN',
# 			'block' : BasicBlock,
# 			'scale_factor' : 8,

# 	},
# 	'DRN58': {
# 			'arch' : 'C',
# 			'layers' : [1, 1, 3, 4, 6, 3, 1, 1],
# 			'type' : 'DRN',
# 			'block' : Bottleneck,
# 			'scale_factor' : 8,

# 	},
# 	'DRN22': {
# 			'arch' : 'D',
# 			'layers' : [1, 1, 2, 2, 2, 2, 1, 1],
# 			'type' : 'DRN',
# 			'block' : BasicBlock,
# 			'scale_factor' : 8,

# 	},
# 	'DRN24': {
# 			'arch' : 'D',
# 			'layers' : [1, 1, 2, 2, 2, 2, 2, 2],
# 			'type' : 'DRN',
# 			'block' : BasicBlock,
# 			'scale_factor' : 8,

# 	},
# 	'DRN38': {
# 			'arch' : 'D',
# 			'layers' : [1, 1, 3, 4, 6, 3, 1, 1],
# 			'type' : 'DRN',
# 			'block' : BasicBlock,
# 			'scale_factor' : 8,

# 	},
# 	'DRN40': {
# 			'arch' : 'D',
# 			'layers' : [1, 1, 3, 4, 6, 3, 2, 2],
# 			'type' : 'DRN',
# 			'block' : BasicBlock,
# 			'scale_factor' : 8,

# 	},
# 	'DRN54': {
# 			'arch' : 'D',
# 			'layers' : [1, 1, 3, 4, 6, 3, 1, 1],
# 			'type' : 'DRN',
# 			'block' : Bottleneck,
# 			'scale_factor' : 8,

# 	},
# 	'DRN56': {
# 			'arch' : 'D',
# 			'layers' : [1, 1, 3, 4, 6, 3, 2, 2],
# 			'type' : 'DRN',
# 			'block' : Bottleneck,
# 			'scale_factor' : 8,

# 	},
# 	'DRN105': {
# 			'arch' : 'D',
# 			'layers' : [1, 1, 3, 4, 23, 3, 1, 1] ,
# 			'type' : 'DRN',
# 			'block' : Bottleneck,
# 			'scale_factor' : 8,

# 	},
# 	'DRN107': {
# 			'arch' : 'D',
# 			'layers' : [1, 1, 3, 4, 23, 3, 2, 2],
# 			'type' : 'DRN',
# 			'block' : Bottleneck,
# 			'scale_factor' : 8,

# 	},
# }


# def make_encoder(e_config, config, pretrained=False):

# 	if e_config['type'] == 'VGG':
# 		model = VGG(make_layers(e_config['layers']), config)

# 	elif e_config['type'] == 'DRN':
# 		model = DRN(e_config['block'], e_config['layers'],
# 				config=config, arch=e_config['arch'])


# 	if pretrained:
# 		sigma = config['dataset']['first_blur_sigma']
# 		e_name = config['model']['name'].split('_')[0]
# 		filename = 'weights/encoder-best-{0}-S{1}.pth.tar'
# 		model.load_weights(filename.format(e_name, sigma))

# 	return model




# class VGG(nn.Module):

# 	def __init__(self, features, config):
# 		super(VGG, self).__init__()

# 		self.features = features
# 		self.config = config
# 		self.classifier = nn.Conv2d(512,1, kernel_size=1)
# 		self.sigmoid = nn.Sigmoid()
# 		# self._initialize_weights()

# 	def forward(self, x, layers=[]):


# 		feat = list()
# 		for name, module in self.features._modules.items():
# 			x = module(x)
# 			if int(name) in layers:
# 				feat.append(x)

# 		x = self.classifier(x)
# 		x = self.sigmoid(x)

# 		return [feat, x]

# 	def _initialize_weights(self):
# 		for m in self.modules():
# 			if isinstance(m, nn.Conv2d):
# 				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# 				m.weight.data.normal_(0, math.sqrt(2. / n))
# 				if m.bias is not None:
# 					m.bias.data.zero_()
# 			elif isinstance(m, nn.BatchNorm2d):
# 				m.weight.data.fill_(1)
# 				m.bias.data.zero_()
# 			elif isinstance(m, nn.Linear):
# 				m.weight.data.normal_(0, 0.01)
# 				m.bias.data.zero_()

# 	def load_weights(self,w_path=''):
# 		if not w_path:
# 			sigma = self.config['dataset']['first_blur_sigma']
# 			e_name = self.config['model']['name'].split('_')[0]
# 			filename = 'weights/encoder-best-{0}-S{1}.pth.tar'
# 			w_path = filename.format(e_name, sigma)

# 		self.load_state_dict(torch.load(w_path)['state_dict'])



# def make_layers(cfg, batch_norm=False):
# 	network = []
# 	in_channels = 3
# 	for box in cfg:
# 		layers = list()
# 		for v in box:
# 			if v == 'M':
# 				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
# 			else:
# 				if 'd' in v:
# 					v = int(''.join(i for i in v if i.isdigit()))
# 					conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
# 				else:
# 					v = int(''.join(i for i in v if i.isdigit()))
# 					conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
# 				if batch_norm:
# 					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
# 				else:
# 					layers += [conv2d, nn.ReLU(inplace=True)]
# 				in_channels = v
# 		network.append(nn.Sequential(*layers))
# 	return nn.Sequential(*network)


# reference : https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch



e_config = {
	'VGG16':{
			'arch' : [['64', '64', 'M'], ['128','128', 'M'], ['256', '256', '256', 'M'],
				['512', '512', '512', 'M'], ['512' , '512', '512', 'M']],
			'scale_factor' : 32
		},
	'DVGG16' :{
			'arch' : [['64', '64', 'M'], ['128','128', 'M'], ['256', '256', '256', 'M'],
				['512', '512', '512'], ['512d', '512d', '512d']],
			'scale_factor' : 8
		}
}


def make_encoder(e_config, config, **kwargs):
	model = Encoder(make_layers(e_config['arch']), config, **kwargs)
	# if pretrained:
	# 	model.load_vgg_weights()
	return model




class Encoder(nn.Module):

	def __init__(self, features, config):
		super(Encoder, self).__init__()

		self.features = features
		self.classifier = nn.Conv2d(512,1, kernel_size=1)
		self.sigmoid = nn.Sigmoid()
		self.config = config
		self._initialize_weights()

	def forward(self, x, layers=range(5)):


		feat = list()
		for name, module in self.features._modules.items():
			x = module(x)
			if int(name) in layers:
				feat.append(x)

		sal = self.classifier(x)

		return [feat, self.sigmoid(sal)]

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


	# def load_weights(self,w_path='weights/encoder-3_1.pth.tar'):
	# 	self.load_state_dict(torch.load(w_path)['state_dict'])

	def load_weights(self,w_path=''):
		if not w_path:
			sigma = self.config['dataset']['first_blur_sigma']
			e_name = self.config['model']['name'].split('_')[0]
			filename = 'weights/encoder-best-{0}-S{1}.pth.tar'
			w_path = filename.format(e_name, sigma)

		self.load_state_dict(torch.load(w_path)['state_dict'])
		print('loading weights {0}'.format(w_path))




def make_layers(cfg, batch_norm=False):
	network = []
	in_channels = 3
	for box in cfg:
		layers = list()
		for v in box:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				if 'd' in v:
					v = int(''.join(i for i in v if i.isdigit()))
					conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
				else:
					v = int(''.join(i for i in v if i.isdigit()))
					conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		network.append(nn.Sequential(*layers))
	return nn.Sequential(*network)

