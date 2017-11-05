
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
lr = .001
keep_prob = 0.8
dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

epoch = 5



def train():
	
	print('building model')
	model = SpatioTemporalSaliency()
        print('initializing')
        model._initialize_weights()
        print("let's bring on gpus!")
        model = model.cuda()

	crit = nn.KLDivLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	# drop = nn.Dropout(keep_prob)
	# hidden_state = model.init_hidden(batch_size)

	print 'prep dataset'

	data = make_batch(size=100)

	start = time.time()
	l = list()
	for ep in xrange(epoch):
		for batch in data:
			images = [img[0] for img in batch]
			images = torch.stack(images)

			seq = np.array([img[1] for img in batch], dtype=np.float32)
			seq_input = torch.from_numpy(seq[:,:-1,...]).unsqueeze(2)
			target = torch.from_numpy(seq).unsqueeze(2)

			images = Variable(images.cuda(), requires_grad=True)
			seq_input = Variable(seq_input.cuda(), requires_grad=True)
			target = Variable(target.cuda(), requires_grad=False)



			output = model(images,seq_input)
			# output = model(im, seq)
			# for i in xrange(input.size(1)-1):
			# 	if i < seq_start:
			# 		out , hidden_state = model(input[:,i,:,:,:].unsqueeze(1), hidden_state)
			# 	else:
			# 		out , hidden_state = model(out, hidden_state)
			# 	output.append(out)

			# output = torch.cat(output,1)
			loss = crit(output, target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			l.append(loss.data[0])

			if batch%10 == 0 and bactch != 0:
				print(np.array(l).mean(), time.time()-start)
				l = list()
				start = time.time()

			# if step%1000 == 0:
			# 	# make video
			# 	print(step)
			# 	print("now generating video!")
			# 	video = cv2.VideoWriter()
			# 	success = video.open("generated_conv_lstm_video_{0}.avi".format(step), fourcc, 4, (180, 180), True)
			# 	hidden_state = model.init_hidden(batch_size)
			# 	output = list()
			# 	for i in xrange(25):
			# 		if i < seq_start:
			# 			out , hidden_state = model(input[:,i,:,:,:].unsqueeze(1), hidden_state)
			# 		else:
			# 			out , hidden_state = model(out, hidden_state)
			# 		output.append(out)
			# 	ims = torch.cat(output,1).permute(0,1,4,3,2)
			# 	ims = ims[0].data.cpu().numpy()
			# 	for i in xrange(5,25):
			# 		x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
			# 		new_im = cv2.resize(x_1_r, (180,180))
			# 		video.write(new_im)
			# 	video.release()




train()

