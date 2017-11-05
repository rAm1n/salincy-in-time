
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from convlstm import ConvLSTM
from encoder import make_encoder
import numpy as np
import cv2
import time



class SpatioTemporalSaliency(nn.Module):

	def __init__(self, num_layers=2):
		super(SpatioTemporalSaliency, self).__init__()
	
		self.encoder = make_encoder(pretrained=False)
		self.CLSTM = ConvLSTM((32,32), 1, [64, 128], [(3,3), (3,3)], num_layers,
				 batch_first=True, bias=True, return_all_layers=True)
		self.conv_out = nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=False)
		
		

	def _init_hidden_state(self):
		return self.CLSTM._init_hidden()

	def forward(self, images, sequence):
		assert images.size()[2:] == (224, 224)
		assert images.size(0) == sequence.size(0)
		features = self.encoder(images)
		out_im, hidden_c = self.CLSTM(features)
		out_seq, hidden_c = self.CLSTM(sequence, hidden_c)

		output = torch.stack([out_im, out_seq], dim=1)
		# out_im_conv = self.conv_out(out_im[-1:])
		# output.append(out_im_conv)
		# for h in out_seq[-1:]:
			# out_seq_conv = self.conv_out(h)
			# output.append(s
		# output.append(out_1_conv)
		return self.conv_out(output)


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
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
		if vgg:
			self.encoder.load_vgg_weights()

