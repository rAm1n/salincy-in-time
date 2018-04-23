
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from layers.decoder import make_decoder, d_config
from layers.encoder import make_encoder, e_config
import numpy as np
import cv2
import time
import os
from config import CONFIG
from dataset import transform
from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
import skimage.transform
from utils import fov_mask



class SpatioTemporalSaliency(nn.Module):
	def __init__(self):
		super(SpatioTemporalSaliency, self).__init__()

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




class RNNSaliency(SpatioTemporalSaliency):   # no batch training support b,c,h,w ---> assert b == 1

	def __init__(self, config):
		super(RNNSaliency, self).__init__()

		self.config = config
		self.build()

	def build(self):
		if self.config['model']['type'] != 'RNN':
			raise ValueError('Model type is not valid.')

		encoder, decoder = self.config['model']['name'].upper().split('_')

		self.encoder = make_encoder(e_config[encoder])
		self.decoder = make_decoder(d_config[decoder])

	def _init_hidden_state(self):
		return self.decoder._init_hidden()

	def forward(self, input, itr=12):

		if self.training:
			images, sal , target, path = input
			t , c, h, w = images.size()
			assert (h, w) == (600, 800)


			result = list()
			hidden_c = None
			features = self.encoder.features(images).data
			for idx in range(t):
				# features = self.encoder.features(images[[idx]])
				feat = Variable(features[[idx]].unsqueeze(1)).cuda()
				# feat_copy = Variable(feat.unsqueeze(1)).cuda()
				output, [_ , hidden_c] = self.decoder(feat, hidden_c)
				result.append(output[0,0])

			return torch.stack(result)


		else: # eval mode --- supports batch?
			images, sal, target, img = input
			t, c, h, w = images.size()
			assert (h, w) == (600, 800)

			img = Image.open(img)


			result = list()
			hidden_c = None
			image = images[[0]]

			for idx in range(itr):
				features = self.encoder.features(image).data
				features = Variable(features.unsqueeze(1), volatile=False).cuda()
				output, [_ , hidden_c] = self.decoder(features, hidden_c)
				result.append(output[0,0])

				image = self._eval_next_frame(img, output[0,0,0].data.cpu().numpy())

				image = Image.fromarray(image)
				image = transform(image)
				image = Variable(image.unsqueeze(0)).cuda(0)

			return torch.stack(result)


	def _eval_next_frame(self, img, mask):
		# prep for next step

		# mask_acc = np.zeros((img.shape[1], img.shape[0]))
		# blurred = np.array(img.filter(ImageFilter.GaussianBlur(self.config['blur_sigma'])))
		# h, w, _ = blurred.shape
		# mask = output[0,0,0].data.cpu().numpy() * 255
		# mask = np.array(Image.fromarray(mask).resize((w,h)))
		# x_max, y_max = np.unravel_index(mask.argmax(), mask.shape)

		# gt = np.zeros((mask.shape[0], mask.shape[1]))
		# gt[x_max, y_max] = 2550
		# gt = gaussian_filter(gt, self.config['gaussian_sigma'])
		# mask = (gt > self.config['mask_th'])
		# blurred[mask] = np.array(img)[mask]


		blurred = np.array(img.filter(ImageFilter.GaussianBlur(self.config['dataset']['blur_sigma'])))
#		mask = output[0,0,0].data.cpu().numpy()
		mask = skimage.transform.resize(mask, img.size[::-1])

		if self.config['eval']['next_frame_policy']=='max':
			y_max, x_max = np.unravel_index(mask.argmax(), mask.shape)
			mask, _ = fov_mask(img.size[::-1], radius=self.config['dataset']['foveation_radius'],
							center=(x_max,y_max), th=self.config['eval']['mask_th'])
		elif self.config['eval']['next_frame_policy'] == 'same':
			mask = (mask > self.config['eval']['mask_th'])

		elif self.config['eval']['next_frame_policy'] =='same_norm':
			mask/= mask.max()
			mask = (mask > self.config['eval']['mask_th'])

		if self.config['eval']['next_frame_policy']=='max_norm':
			mask/=mask.max()
			y_max, x_max = np.unravel_index(mask.argmax(), mask.shape)
			mask, _ = fov_mask(img.size[::-1], radius=self.config['dataset']['foveation_radius'],
							center=(x_max,y_max), th=self.config['eval']['mask_th'])

		blurred[mask] = np.array(img)[mask]

		return blurred

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

