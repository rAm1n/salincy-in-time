# code reference : https://github.com/ndrplz/ConvLSTM_pytorch

import torch.nn as nn
from torch.autograd import Variable
import torch




d_config = {
	'CLSTM1-32':{
			'input_dim' : 512,
			'hidden_dims' : [32],
			'kernels' : [(3,3)],
			'bidirectional': False,
			'cell' : 'ConvLSTMCell',
		},

	'ATTNCLSTM1-32':{
			'input_dim' : 512,
			'hidden_dims' : [32],
			'kernels' : [(3,3)],
			'bidirectional': False,
			'cell' : 'AttnConvLSTMCell'
		},

	'CLSTM1-64':{
			'input_dim' : 512,
			'hidden_dims' : [64],
			'kernels' : [(3,3)],
			'bidirectional': False,
			'cell' : 'ConvLSTMCell',
		},

	'ATTNCLSTM1-64':{
			'input_dim' : 512,
			'hidden_dims' : [64],
			'kernels' : [(3,3)],
			'bidirectional': False,
			'cell' : 'AttnConvLSTMCell'
		},

	'ATTNCLSTM1-256':{
			'input_dim' : 512,
			'hidden_dims' : [256],
			'kernels' : [(3,3)],
			'bidirectional': False,
			'cell' : 'AttnConvLSTMCell'
		},

	'CLSTM2':{
			'input_dim' : 512,
			'hidden_dims' : [64 , 64],
			'kernels' : [(3,3), (3,3)],
			'bidirectional': False,
			'cell' : 'ConvLSTMCell',
		},
	'CLSTM4':{
			'input_dim' : 512,
			'hidden_dims' : [64 , 64, 64 , 64],
			'kernels' : [(3,3), (3,3), (3,3), (3,3)],
			'bidirectional': False,
			'cell' : 'ConvLSTMCell',
		},
	'BCLSTM3':{
			'input_dim' : 512,
			'hidden_dims' : [64 , 64, 64],
			'kernels' : [(3,3), (3,3), (3,3)],
			'bidirectional': True,
			'cell' : 'ConvLSTMCell',
		},

	'ATTNCLSTM3-64':{
			'input_dim' : 512,
			'hidden_dims' : [64, 64, 64],
			'kernels' : [(3,3), (3,3), (3,3)],
			'bidirectional': False,
			'cell' : 'AttnConvLSTMCell'
		},

	'ATTNBCLSTM3-64':{
			'input_dim' : 512,
			'hidden_dims' : [64 , 64, 64],
			'kernels' : [(3,3), (3,3), (3,3)],
			'bidirectional': True,
			'cell' : 'AttnConvLSTMCell',
		},
}



def make_decoder(config):
	return Custom_ConvLstm(config)





class ChannelSoftmax(nn.Module):
	def __init__(self):
		super(ChannelSoftmax, self).__init__()
		self.softmax = nn.Softmax()
	def forward(self, input_):
		b,c,h,w = input_.size()
		output_ = torch.stack([self.softmax(input_[:,i].contiguous().view(b,-1)) for i in range(c)], 0)
		return output_.view(b,c,h,w)



class ConvLSTMCell(nn.Module):

	def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
		"""
		Initialize ConvLSTM cell.

		Parameters
		----------
		input_size: (int, int)
			Height and width of input tensor as (height, width).
		input_dim: int
			Number of channels of input tensor.
		hidden_dim: int
			Number of channels of hidden state.
		kernel_size: (int, int)
			Size of the convolutional kernel.
		bias: bool
			Whether or not to add the bias.
		"""

		super(ConvLSTMCell, self).__init__()

		self.height, self.width = input_size
		self.input_dim  = input_dim
		self.hidden_dim = hidden_dim

		self.kernel_size = kernel_size
		self.padding = kernel_size[0] // 2, kernel_size[1] // 2
		self.bias = bias

		self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
							  out_channels=4 * self.hidden_dim,
							  kernel_size=self.kernel_size,
							  padding=self.padding,
							  bias=self.bias)

	def forward(self, input_tensor, cur_state):

		h_cur, c_cur = cur_state
		combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

		combined_conv = self.conv(combined)
		cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
		i = torch.sigmoid(cc_i)
		f = torch.sigmoid(cc_f)
		o = torch.sigmoid(cc_o)
		g = torch.tanh(cc_g)

		c_next = f * c_cur + i * g
		h_next = o * torch.tanh(c_next)

		return (h_next, c_next)

	def init_hidden(self, batch_size):
		return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
				Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())




class AttnConvLSTMCell(ConvLSTMCell):

	def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
		"""
		Initialize ConvLSTM cell.

		Parameters
		----------
		input_size: (int, int)
			Height and width of input tensor as (height, width).
		input_dim: int
			Number of channels of input tensor.
		hidden_dim: int
			Number of channels of hidden state.
		kernel_size: (int, int)
			Size of the convolutional kernel.
		bias: bool
			Whether or not to add the bias.
		"""

		super(ConvLSTMCell, self).__init__()


		self.attn_conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
							  out_channels=self.hidden_dim,
							  kernel_size=self.kernel_size,
							  padding=self.padding,
							  bias=self.bias)

		self.v_conv = nn.Conv2d(in_channels=self.hidden_dim,
							  out_channels=self.input_dim,
							  kernel_size=self.kernel_size,
							  padding=self.padding,
							  bias=self.bias)

		self.softmax = nn.Softmax2d()
		# self.activation = torch.tanh
		self.activation = torch.tanh


	def forward(self, input_tensor, cur_state):

		# attention
		h_cur, c_cur = cur_state

		combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
		ZT = self.v_conv(self.activation(self.attn_conv(combined)))
		At = self.softmax(ZT)
		input_ = At * input_tensor


		# Regular ConvLSTM
		conv_input = torch.cat([input_, h_cur], dim=1)
		combined_conv = self.conv(conv_input)
		cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
		i = torch.sigmoid(cc_i)
		f = torch.sigmoid(cc_f)
		o = torch.sigmoid(cc_o)
		g = torch.tanh(cc_g)

		c_next = f * c_cur + i * g
		h_next = o * torch.tanh(c_next)

		return (h_next, c_next)



class ConvLSTM(nn.Module):
	"""
		if bidirectional is True, the number of layers must be odd.

	"""

	def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
				 bidirectional=False, batch_first=False, bias=True, return_all_layers=False, cell=ConvLSTMCell):
		super(ConvLSTM, self).__init__()

		self._check_kernel_size_consistency(kernel_size)

		# Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
		kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
		hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
		if not len(kernel_size) == len(hidden_dim) == num_layers:
			raise ValueError('Inconsistent list length.')
		if bidirectional and ((num_layers%2) == 0 ):
			raise ValueError('only supports odd number of layers for bidirectional ConvLSTM')

		self.height, self.width = input_size

		self.input_dim  = input_dim
		self.hidden_dim = hidden_dim
		self.kernel_size = kernel_size
		self.num_layers = num_layers
		self.batch_first = batch_first
		self.bias = bias
		self.bidirectional = bidirectional
		self.return_all_layers = return_all_layers

		cell_list = []
		for i in range(0, self.num_layers):
			cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

			cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
										  input_dim=cur_input_dim,
										  hidden_dim=self.hidden_dim[i],
										  kernel_size=self.kernel_size[i],
										  bias=self.bias))

		self.cell_list = nn.ModuleList(cell_list)

	def forward(self, input_tensor, hidden_state=None):
		"""

		Parameters
		----------
		input_tensor: todo
			5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
		hidden_state: todo
			None. todo implement stateful

		Returns
		-------
		last_state_list, layer_output
		"""
		if not self.batch_first:
			# (t, b, c, h, w) -> (b, t, c, h, w)
			input_tensor.permute(1, 0, 2, 3, 4)

		# Implement stateful ConvLSTM
		if hidden_state is not None:
			# raise NotImplementedError()
			pass
		else:
			hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

		layer_output_list = []
		last_state_list   = []

		seq_len = input_tensor.size(1)
		cur_layer_input = input_tensor

		for layer_idx in range(self.num_layers):


			h, c = hidden_state[layer_idx]
			output_inner = []
			for t in range(seq_len):
				h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
												 cur_state=[h, c])
				output_inner.append(h)

			if self.bidirectional:
				output_inner = output_inner[::-1]

			layer_output = torch.stack(output_inner, dim=1)
			cur_layer_input = layer_output

			layer_output_list.append(layer_output)
			last_state_list.append([h, c])


		if not self.return_all_layers:
			layer_output_list = layer_output_list[-1:]
			last_state_list   = last_state_list[-1:]

		return layer_output_list, last_state_list

	def _init_hidden(self, batch_size):
		init_states = []
		for i in range(self.num_layers):
			init_states.append(self.cell_list[i].init_hidden(batch_size))
		return init_states

	@staticmethod
	def _check_kernel_size_consistency(kernel_size):
		if not (isinstance(kernel_size, tuple) or
					(isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
			raise ValueError('`kernel_size` must be tuple or list of tuples')

	@staticmethod
	def _extend_for_multilayer(param, num_layers):
		if not isinstance(param, list):
			param = [param] * num_layers
		return param






class Custom_ConvLstm(nn.Module):

	def __init__(self, config):
		super(Custom_ConvLstm, self).__init__()

		self.CLSTM = ConvLSTM((75,100), config['input_dim'], config['hidden_dims'],
				config['kernels'], len(config['hidden_dims']), batch_first=True,
				bias=True, return_all_layers=True,  cell=eval(config['cell']))

		self.conv_out = nn.Conv2d(config['hidden_dims'][-1], 1, kernel_size=3, padding=1, bias=False)
		self.sigmoid = nn.Sigmoid()


	def forward(self, input, hidden_c=None):
		_b, _t, _c , _h, _w = input.size()
		assert (_t, _c, _h, _w) == (1 , 512 , 75, 100)

		conv_output = list()
		output, hidden_c = self.CLSTM(input, hidden_c)
		output = output[-1]
		for t in range(output.size(1)):
			conv1_1_out = self.conv_out(output[:,t,...])
			b , c, h, w = conv1_1_out.size()
			conv_output.append(self.sigmoid(conv1_1_out.view(b,-1)).view(b,c,h,w))
			# conv_output.append(self.sigmoid(conv1_1_out))
			# conv_output.append(conv1_1_out)

		conv_output = torch.stack(conv_output, dim=1)
		return conv_output, [output, hidden_c]



