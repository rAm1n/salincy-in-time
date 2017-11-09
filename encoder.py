# reference : https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math



model_urls = {
	'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
	'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
	'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
	'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
	'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
	'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
	'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
	'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']

class Encoder(nn.Module):

	def __init__(self, features, num_classes=32*32):
		super(Encoder, self).__init__()
		self.features = features
		out_channels = self.features[-3].out_channels
		self.num_classes = num_classes
		self.embedding = nn.Sequential(
			nn.Linear(out_channels * 28 * 28, num_classes),
#			nn.ReLU(True),
#			nn.Dropout(),
#			nn.Linear(4096, 4096),
#			nn.ReLU(True),
#			nn.Dropout(),
#			nn.Linear(4096, num_classes),
		)
#		self._initialize_weights()

	def forward(self, input):
		x = self.features(input)
		x = x.view(x.size(0), -1)
		x = self.embedding(x)
		h = w = int(math.sqrt(self.num_classes))
		return x.view(x.size(0), 1, 1, h, w)

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

	def load_vgg_weights(self, url='https://download.pytorch.org/models/vgg16-397923af.pth'):
		vgg = model_zoo.load_url(url)
		for idx,m in enumerate(self.features.modules()):
			idx = idx - 1
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data = vgg.get('features.{0}.weight'.format(idx))
				if m.bias is not None:
					m.bias.data = vgg.get('features.{0}.bias'.format(idx))




def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)



def make_encoder(pretrained=False, config=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'], **kwargs):
	model = Encoder(make_layers(config), **kwargs)
	if pretrained:
		model.load_vgg_weights()
	return model

