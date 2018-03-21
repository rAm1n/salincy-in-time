# reference : https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch



en_config = {
	'vgg16': ['64', '64', 'M', '128','128', 'M', '256', '256', '256', 'M',
				'512', '512', '512', 'M', '512' , '512', '512', 'M'],
	'dvgg16' : ['64', '64', 'M', '128','128', 'M', '256', '256', '256', 'M',
				'512', '512', '512', '512d', '512d', '512d']
}






class Encoder(nn.Module):

	def __init__(self, features):
		super(Encoder, self).__init__()

		self.features = features

	def forward(self, x):

		return self.features(x)

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

	# def load_vgg_weights(self, url='https://download.pytorch.org/models/vgg16-397923af.pth'):
	# 	vgg = model_zoo.load_url(url)
	# 	for idx,m in enumerate(self.features.modules()):
	# 		idx = idx - 1
	# 		if isinstance(m, nn.Conv2d):
	# 			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
	# 			m.weight.data = vgg.get('features.{0}.weight'.format(idx))
	# 			if m.bias is not None:
	# 				m.bias.data = vgg.get('features.{0}.bias'.format(idx))



def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
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
	return nn.Sequential(*layers)



def make_encoder(config, pretrained=False, **kwargs):
	model = Encoder(make_layers(config), **kwargs)
	if pretrained:
		model.load_vgg_weights()
	return model



# class Encoder(nn.Module):

# 	def __init__(self, features):
# 		super(Encoder, self).__init__()

# 		self.features = features
# 		out_channels = self.features[-3].out_channels

# 		self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
# 		self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')

#         #############################################################

# 		self.embedding = nn.Sequential(
# 			nn.Conv2d(896, 448, kernel_size=(3,3), stride=1,padding=1),
# 			nn.AvgPool2d(kernel_size=(2, 2), stride=2),
# 			nn.Conv2d(448, 32,  kernel_size=(3,3), stride=1,padding=1),
# 			nn.Conv2d(32, 1, kernel_size=(3,3), stride=1, padding=1),
# 		)

# 		##############################################################
# 	def forward(self, x, extracted_layers=['9', '16', '23']):
# 		outputs = []
# 		for name, module in self.features._modules.items():
# 			x = module(x)
# 			if name in extracted_layers:
# 				outputs += [x]

# 		merge = torch.cat([outputs[0], self.up2(outputs[1]), self.up4(outputs[2])], dim=1)
# 		return self.embedding(merge)

# 		# x = self.features(input)
# 		# return x
# #		x = x.view(x.size(0), -1)
# #		x = self.embedding(x)
# #		h = w = int(math.sqrt(self.num_classes))
# #		return x.view(x.size(0), 1, 1, h, w)

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

# 	def load_vgg_weights(self, url='https://download.pytorch.org/models/vgg16-397923af.pth'):
# 		vgg = model_zoo.load_url(url)
# 		for idx,m in enumerate(self.features.modules()):
# 			idx = idx - 1
# 			if isinstance(m, nn.Conv2d):
# 				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# 				m.weight.data = vgg.get('features.{0}.weight'.format(idx))
# 				if m.bias is not None:
# 					m.bias.data = vgg.get('features.{0}.bias'.format(idx))







