
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from convlstm import ConvLSTM, weights_init
from encoder import make_encoder
import numpy as np
import cv2
import time



class SpatioTemporalSaliency(nn.Module):

	def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers):

		self.encoder = make_encoder(pretrained=True)
		self.CLSTM = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers,
				 batch_first=False, bias=True, return_all_layers=False)

	def _init_hidden_state(self):
		return self.CLSTM._init_hidden()

	def forward(input):
		assert input.size()[2:] == (224, 224)
		features = self.encoder(input)

