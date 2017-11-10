
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

	def __init__(self, num_layers=2):
		super(SpatioTemporalSaliency, self).__init__()

		self.encoder = make_encoder(pretrained=False)
		self.Custom_CLSTM = Custom_ConvLstm()


	def _init_hidden_state(self):
		return self.CLSTM._init_hidden()

	def forward(self, images, sequence=None ,itr=30):
		assert images.size()[2:] == (224, 224)

		features = self.encoder(images)
		out_im, lstm_out = self.Custom_CLSTM(features)

		if not (sequence is None):
			assert images.size(0) == sequence.size(0)
			out_seq, lstm_out = self.Custom_CLSTM(sequence, lstm_out[1])
			tmp = torch.cat((out_im, out_seq), dim=1)
			out = list()
			for t in xrange(tmp.size(1)):
				out.append(F.sigmoid(tmp[:,t,...]))
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
			out = list()
			for t in range(tmp.size(1)):
				out.append(F.sigmoid(tmp[:,t,...]))
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
				m.weight.data.normal_(0, 0.1)
				m.bias.data.zero_()
		if vgg:
			self.encoder.load_vgg_weights()

