
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
from PIL import Image
import torchvision.transforms as transforms
import skvideo.io


batch_size=4
im_size=(256,256)
size=5000
gamma = 3

act= None
opt='ADAM'
fine_tune=False
lr=0.0005
B1=0.01
B2=0.999
eps=1e-5
epoch = 5

ck_path = '/media/ramin/monster/models/sequence-kld'
vid_path = 'video-kld'

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')


img_processor = transforms.Compose([
	transforms.Scale(im_size),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	]
)



def train():

	print('building model')
	model = SpatioTemporalSaliency(activation=act)
	print('initializing')
	model._initialize_weights(True)
	print("let's bring on gpus!")
	model = model.cuda()


	if model.activation == 'softmax':
		crit = nn.KLDivLoss()
	elif model.activation == 'sigmoid':
		crit = nn.MSELoss()
	else:
		crit = nn.MSELoss()

	if opt == 'ADAM':
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(B1, B2), eps=eps)
	elif opt=='SGD':
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


	if not fine_tune:
		for param in model.encoder.features.parameters():
				param.requires_grad = False


	print('prep dataset')

	d = Salicon(size=size, gamma=gamma)
	d.load()
	
	start = time.time()
	l = list()
	for ep in range(epoch):
		iteration = len(d._map['train']) // batch_size
		for step in range(iteration):
			if act=='softmax':
				batch = d.next_batch(batch_size, mode='train', norm='L1')
			else:
				batch = d.next_batch(batch_size, mode='train', norm=None)

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
				#torch.cuda.synchronize()
			except Exception as x:
				print(x)
				return 0


			if step%20 == 0 and step != 0:
				print('epoch {0} , step {1} / {2}   , mean-loss: {3}  ,  time: {4}'.format(ep, step, iteration, np.array(l).mean(), time.time()-start))
				l = list()
				start = time.time()

			if step%100 == 0 and step !=0:
				try:
					#make video
					print(step)
					print("now generating video!")
					video = cv2.VideoWriter()
					#success = video.open("{0}/generated_conv_lstm_video_{1}_{2}.avi".format(vid_path, ep, step), fourcc, 4, im_size, True)
					writer = skvideo.io.FFmpegWriter("{0}/generated_conv_lstm_video_{1}_{2}.avi".format(vid_path, ep, step))
					model.eval()
					img = d.next_batch(batch_size=1, mode='test')[0][0]
					img_resize = img.resize(im_size)
					img_processed = Variable(img_processor(img).unsqueeze(0).cuda())

					output = model(img_processed, sequence=None, itr=64)
					output = output.permute(0,1,4,3,2)
					ims = output[0].squeeze().data.cpu().numpy()
					print(ims.shape)
					for i in range(ims.shape[0]):
						if act=='sigmoid':
							x_1_r = np.uint8((np.maximum(ims[i], 0)) * 255)
						elif act=='softmax':
							x_1_r = np.uint8((np.maximum(ims[i], 0) / ims[i].max()) * 255)
						else:
							x_1_r = np.uint8(np.minimum(np.maximum(ims[i], 0), 255)) 
							x_1_r = ((x_1_r - x_1_r.min()) / (x_1_r.max() - x_1_r.min())) * 255.0

						mask = Image.fromarray(x_1_r).resize(im_size).convert('RGB')
						new_im = Image.blend(img_resize, mask, alpha=0.7).convert('RGB')
						#new_im = cv2.resize(x_1_r, (256,256))
						#video.write(np.asarray(new_im))
						writer.writeFrame(new_im)
					#video.release()
					writer.close()
					model.train()
	
				except Exception as x:
					print(x)

			if step%20000 == 0:
				model.save_checkpoint(model.state_dict(), ep, step, path=ck_path)
				
			del images, seq_input, target, loss



train()

