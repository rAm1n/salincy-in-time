
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from convlstm import Custom_ConvLstm
from encoder import make_encoder
import numpy as np
import cv2
import time



class SpatioTemporalSaliency(nn.Module):

	def __init__(self, num_layers=1, grid= 32 ):
		super(SpatioTemporalSaliency, self).__init__()

		self.encoder = make_encoder(pretrained=False)
		self.Custom_CLSTM = Custom_ConvLstm()
		self.img_embedding = nn.Linear(256 * 28 * 28, grid * grid)
		self.seq_embedding = nn.Linear(grid*grid,  grid*grid)
		self.grid = grid
	
	def _init_hidden_state(self):
		return self.CLSTM._init_hidden()

	def forward(self, images, sequence=None ,itr=30):
		assert images.size()[2:] == (224, 224)
		b , c, h, w = images.size()
		features = self.encoder(images).view(b, -1)
		img_emd = self.img_embedding(features)
		img_emd = img_emd.view(b,1,1, self.grid, self.grid)
		out_im, lstm_out = self.Custom_CLSTM(img_emd)

		if not (sequence is None):
			assert images.size(0) == sequence.size(0)
			b , t, c, h, w = sequence.size()
			seq_tmp = list()
			for i in xrange(t):
				seq_tmp.append(self.seq_embedding(sequence[:,i,...].contiguous().view(b,-1)).contiguous().view(b,c,h,w))
			seq_emd = torch.stack(seq_tmp, 1)
#			seq_emd = self.seq_embedding(sequence.view(b,-1)).view(b, t, c, h, w)
			out_seq, lstm_out = self.Custom_CLSTM(seq_emd, lstm_out[1])
#			out_seq, lstm_out = self.Custom_CLSTM(sequence, lstm_out[1])
			tmp = torch.cat((out_im, out_seq), dim=1)
			out = list()
			b , t, c, h , w = tmp.size()
			for t in xrange(tmp.size(1)):
				q = F.log_softmax(tmp[:,t,...].contiguous().view(b, -1))
				out.append(q.view(b,c,h,w))
				#out.append(F.sigmoid(tmp[:,t,...]))
			result = torch.stack(out, dim=1)
			
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
			out = list()
			for t in range(tmp.size(1)):
				out.append(F.softmax(tmp[:,t,...].contiguous().view(b,-1)).view(b,c,h,w))
			result = torch.stack(out, dim=1)

		return result

	def _initialize_weights(self,vgg=True):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, np.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 1)
				m.bias.data.zero_()
		if vgg:
			self.encoder.load_vgg_weights()

