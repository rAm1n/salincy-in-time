
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from model import SpatioTemporalSaliency
import numpy as np
import cv2
import time
from utils.data import make_batch
import gc
import scipy.misc


num_features=10
filter_size=3
batch_size=4
shape=(32,32) #H,W
inp_chans=3
nlayers=1
seq_len=10
num_balls = 2
max_step = 200000
seq_start = 5
lr = 0.0001
keep_prob = 0.8
dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

epoch = 5

test = scipy.misc.imread('test.jpg')
test = Variable(torch.from_numpy(test)).permute(2,0,1).unsqueeze(0).unsqueeze(0)

def train():

	print('building model')
	model = SpatioTemporalSaliency(num_layers=1)
	print('initializing')
	model._initialize_weights()
	print("let's bring on gpus!")
	model = model.cuda()

	crit = nn.KLDivLoss()
	#crit = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	# drop = nn.Dropout(keep_prob)
	# hidden_state = model.init_hidden(batch_size)

	print 'prep dataset'

	data = make_batch(size=200, batch_size=4)

	start = time.time()
	l = list()
	for ep in xrange(epoch):
		for step, batch in enumerate(data):
			images = [img[0] for img in batch]
			images = torch.stack(images)

			seq = np.array([img[1] for img in batch], dtype=np.float32)
			seq_input = torch.from_numpy(seq[:,1:,...]).unsqueeze(2)
			target = torch.from_numpy(seq).unsqueeze(2)

			images = Variable(images.cuda(), requires_grad=True)
			seq_input = Variable(seq_input.cuda(), requires_grad=True)
			target = Variable(target.cuda(), requires_grad=False)

			#torch.cuda.synchronize()
			output = model(images, seq_input)
			loss = 0
			for t in xrange(target.size(1)):
				loss += crit(output[:,t,...].squeeze(), target[:,t,...].squeeze())

			try:
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				l.append(loss.data[0])
				torch.cuda.synchronize()
			except Exception,x:
				print x
				return 0


			if step%10 == 0 and step != 0:
				print(step, np.array(l).mean(), time.time()-start)
				l = list()
				start = time.time()

			if step%100 == 0:
				try:
					#make video
					print(step)
					print("now generating video!")
					video = cv2.VideoWriter()
					success = video.open("video/generated_conv_lstm_video_{0}.avi".format(step), fourcc, 4, (32, 32), False)
				# 	hidden_state = model.init_hidden(batch_size)
					model.eval()
					output = model(images[0].unsqueeze(0), sequence=None, itr=10)
					#output = model(test , sequence=None, itr=10)
					output = output.permute(0,1,4,3,2)
					global ims
					ims = output[0].data.cpu().numpy()
					for i in xrange(ims.shape[0]):
						x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
						new_im = cv2.resize(x_1_r, (32,32))
						video.write(new_im)
					video.release()
						

					model.train()
				except Exception,x:
					print x
			del images, seq_input, target, loss



train()

