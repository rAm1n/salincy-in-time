
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from convlstm import Custom_ConvLstm
from encoder import make_encoder, en_config
import numpy as np
import cv2
import time
import os
from config import CONFIG
from dataset import transform
from PIL import Image, ImageFilter

class SpatioTemporalSaliency(nn.Module):   # no batch training support b,c,h,w ---> assert b == 1

	def __init__(self, config):
		super(SpatioTemporalSaliency, self).__init__()


		self.config = config
		self.encoder = make_encoder(en_config[config['encoder']]['arch']).cuda(0)
		self.Custom_CLSTM = Custom_ConvLstm().cuda(1)

	def _init_hidden_state(self):
		return self.CLSTM._init_hidden()

	def forward(self, x, itr=8):

		if self.training:
			images, sal , target, path = x
			t , c, h, w = images.size()
			assert (h, w) == (600, 800)


			result = list()
			hidden_c = None
			features = self.encoder.features(images)
			for idx in range(t):
				# features = self.encoder.features(images[[idx]])
				feat = features[[idx]]
				feat_copy = Variable(feat.data.unsqueeze(1)).cuda(1)
				output, [_ , hidde_c] = self.Custom_CLSTM(feat_copy, hidden_c)
				result.append(output[0,0])
			return torch.stack(result)


		else:  # eval mode --- supports batch?
			images, sal, target, img = x
			t, c, h, w = images.size()
			assert (h, w) == (600, 800)
			img = Image.open(img)


			result = list()
			hidden_c = None
			image = images[[0]]
			for idx in range(itr):
				features = self.encoder.features(image)
				feat_copy = Variable(features.data.unsqueeze(1)).cuda(1)
				output, [_ , hidde_c] = self.Custom_CLSTM(feat_copy, hidden_c)
				result.append(output[0,0])

				# prep for next step
				
				blurred = np.array(img.filter(ImageFilter.GaussianBlur(self.config['blur_sigma'])))
				mask = np.array(Image.fromarray(output[0,0,0].data.cpu().numpy() * 255).resize((800,600))) / 255.0
				mask = (mask > self.config['test_mask_th'])
				blurred[mask] = np.array(img)[mask]
				image = Image.fromarray(blurred)
				image = transform(image)
				image = Variable(image.unsqueeze(0)).cuda(0)

			return torch.stack(result)



	def _initialize_weights(self,pretrained=True):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, np.sqrt(4. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				#m.weight.data.normal_(0, 1)
				torch.nn.init.xavier_uniform(m.weight.data)
				m.bias.data.zero_()
		if pretrained:
			self.encoder.load_weights()


	def save_checkpoint(self, state, ep, step, max_keep=15, path='/media/ramin/monster/models/sequence/'):
		filename = os.path.join(path, 'ck-{0}-{1}.pth.tar'.format(ep, step))
		torch.save(state, os.path.join(path, 'ck-last.path.tar'))
		torch.save(state, filename)
		def sorted_ls(path):
			mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
			return list(sorted(os.listdir(path), key=mtime))
		files = sorted_ls(path)[:-max_keep]
		for item in files:
			os.remove(os.path.join(path, item))

	def load_checkpoint(self, path='/media/ramin/monster/models/sequence/', filename=None):
		if not filename:
			filename = os.path.join(path, 'ck-last.path.tar')
		else:
			filename = os.path.join(path, filename)

		self.load_state_dict(torch.load(filename))

