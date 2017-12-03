
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from layers.convlstm import Custom_ConvLstm
from layers.encoder import make_encoder
import numpy as np
import cv2
import time
import os


class SpatioTemporalSaliency(nn.Module):

	def __init__(self, grid= 32, activation='sigmoid'):
		super(SpatioTemporalSaliency, self).__init__()

		self.encoder = make_encoder(pretrained=False)
		self.Custom_CLSTM = Custom_ConvLstm()
		#self.img_embedding = nn.nn.Linear(512 * 14 * 14, grid * grid)
		self.img_embedding = nn.Sequential(nn.Linear( 512 * 8 * 8, grid * grid), nn.Dropout(.75))
		# self.img_embedding = nn.Sequential(nn.Linear(32*32, 32*32), nn.Dropout(0.75))
		self.seq_embedding = nn.Sequential(nn.Linear( grid * grid, grid * grid), nn.Dropout(.75))

		self.grid = grid
		self.activation = activation
	
	def _init_hidden_state(self):
		return self.CLSTM._init_hidden()

	def forward(self, images, sequence=None ,itr=30):
		assert images.size()[2:] == (256, 256)

		b , c, h, w = images.size()
		features = self.encoder(images).view(b, -1)
		img_emd = self.img_embedding(features).view(b ,1 ,1 ,self.grid, self.grid)
		# img_emd = features.view(b,1,1,self.grid, self.grid)

		out_im, lstm_out = self.Custom_CLSTM(img_emd)

		if not (sequence is None):  # training mode
			assert images.size(0) == sequence.size(0)
			b , t, c, h, w = sequence.size()
			# re-weighting sequence inputs
			seq_emd = list()
			for i in range(t):
				seq_emd.append(self.seq_embedding(sequence[:,i,...].contiguous().view(b,-1)).contiguous().view(b,c,h,w))
			seq_emd = torch.stack(seq_emd, 1)

			# running LSTM model for sequences
			out_seq, lstm_out = self.Custom_CLSTM(seq_emd, lstm_out[1])
			out_seq = torch.cat((out_im, out_seq), dim=1)
			# applying softmax at the end.

			result = list()
			b , t, c, h, w = out_seq.size()
			for t in range(out_seq.size(1)):
				if self.activation=='softmax':
					result.append(F.log_softmax(out_seq[:,t,...].contiguous().view(b,-1)).view(b,c,h,w))
				elif self.activation=='sigmoid':
					result.append(F.sigmoid(out_seq[:,t,...].contiguous().view(-1)).view(b,c,h,w))
				else:
					result.append(out_seq[:,t,...])
			result = torch.stack(result, dim=1)
			# result = out_seq

			
		else:

			out_seq = list()
			tmp_out = out_im
			hidden_c = [None, None]
			for t in range(itr):
				tmp_out , hidden_c = self.Custom_CLSTM(tmp_out , hidden_c[1])
				out_seq.append(tmp_out)
			out_seq = torch.cat(out_seq, dim=1)		
			tmp = torch.cat((out_im ,  out_seq), dim=1)
			b , t, c, h , w = tmp.size()
			result = list()
			for t in range(tmp.size(1)):
				if self.activation=='softmax':
					result.append(F.softmax(tmp[:,t,...].contiguous().view(b,-1)).view(b,c,h,w))
				elif self.activation=='sigmoid':
					result.append(F.sigmoid(tmp[:,t,...].contiguous().view(-1)).view(b,c,h,w))
				else:
					result.append(tmp[:,t,...])
			result = torch.stack(result, dim=1)
			#result = tmp

		return result

	def _initialize_weights(self,vgg=True):
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
		if vgg:
			self.encoder.load_vgg_weights()
					

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
		
