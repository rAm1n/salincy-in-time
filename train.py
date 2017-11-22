
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from model import SpatioTemporalSaliency
import numpy as np
import cv2
import time
from utils.salicon import Salicon
import gc
import scipy.misc


num_features=10
filter_size=3
batch_size=8
lr=0.0005
B1=0.01
B2=0.999
eps=1e-5
dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

epoch = 5

def train():

	print('building model')
	model = SpatioTemporalSaliency(num_layers=1)
	print('initializing')
	model._initialize_weights(True)
	print("let's bring on gpus!")
	model = model.cuda()



	crit = nn.KLDivLoss()
	#crit = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(B1, B2), eps=eps)
	#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
	for param in model.encoder.parameters():
			param.requires_grad = False


	print('prep dataset')

	d = Salicon()
	d.load()
	
	start = time.time()
	l = list()
	for ep in range(epoch):
		iteration = len(d._map['train']) // batch_size
		for step in range(iteration):
			batch = d.next_batch(batch_size)
			images = [img[0] for img in batch]
			images = torch.stack(images)

			seq = np.array([img[1] for img in batch], dtype=np.float32)
			seq_input = torch.from_numpy(seq[:,:-1,...]).unsqueeze(2)
			#target = torch.from_numpy(seq[:,1:,...]).unsqueeze(2)
			target = torch.from_numpy(seq).unsqueeze(2)			

			images = Variable(images.cuda(), requires_grad=True)
			seq_input = Variable(seq_input.cuda(), requires_grad=True)
			target = Variable(target.cuda(), requires_grad=False)

			#torch.cuda.synchronize()
			output = model(images, seq_input)
			loss = 0
			for t in range(target.size(1)):
				loss += crit(output[:,t,...].squeeze(), target[:,t,...].squeeze())

			try:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				l.append(loss.data[0])
				torch.cuda.synchronize()
			except Exception as x:
				print(x)
				return 0


			if step%20 == 0 and step != 0:
				print('epoch {0} , step {1} / {2}   , mean-loss: {3}  ,  time: {4}'.format(ep, step, iteration, np.array(l).mean(), time.time()-start))
				l = list()
				start = time.time()

			if step%1000 == 0 and step !=0:
				try:
					#make video
					print(step)
					print("now generating video!")
					video = cv2.VideoWriter()
					success = video.open("video-kld/generated_conv_lstm_video_{0}_{1}.avi".format(ep, step), fourcc, 4, (224, 224), False)
				# 	hidden_state = model.init_hidden(batch_size)
					model.eval()
					output = model(images[0].unsqueeze(0), sequence=None, itr=64)
					#output = model(test , sequence=None, itr=10)
					output = output.permute(0,1,4,3,2)
					global ims
					ims = output[0].data.cpu().numpy()
					for i in range(ims.shape[0]):
						x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
						new_im = cv2.resize(x_1_r, (224,224))
						video.write(new_im)
					video.release()
					model.train()
	
				except Exception as x:
					print(x)

			if step%20000 == 0:
				model.save_checkpoint(model.state_dict(), ep, step, path='/media/ramin/monster/models/sequence-kld/')
				
			del images, seq_input, target, loss



train()

