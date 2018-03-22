from __future__ import print_function
import argparse
import os
import shutil
import time
import numpy as np
import sys
from datetime import datetime
import logging
import cv2
import time
import scipy.misc
from PIL import Image
import gc
import skvideo.io


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model import SpatioTemporalSaliency
from dataset import SequnceDataset
from config import CONFIG



# from utils.salicon import Salicon


# from dataset.loader import Dataset
# from models import *
# from utils import config

parser = argparse.ArgumentParser(description='Scanpath prediction')

parser.add_argument('--weights', default='weights', metavar='DIR',
					help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='dvgg16',
					choices=['vgg16', 'dvgg16'],
					help='model architecture: ' +
						' (default: dvgg16)')
parser.add_argument('--log', default='logs/', metavar='DIR',
					help='path to dataset')
parser.add_argument('--metric', '-m', metavar='METRIC', default='AUC',
					choices=['AUC', 'NSS'],
					help='evaluation metric')
parser.add_argument('-v','--visualize', metavar='DIR', 
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
# 					help='use pre-trained model')



dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')



best_prec1 = 0


def main():
	global args, best_prec1, model, train_dataset, val_dataset
	args = parser.parse_args()


	logging.basicConfig(
		format="%(message)s",
		handlers=[
			logging.FileHandler("{0}/{1}.log".format(args.log, sys.argv[0].replace('.py','') + datetime.now().strftime('_%H_%M_%d_%m_%Y'))),
			logging.StreamHandler()
		],
		level=logging.INFO)

	# create model
	# if args.pretrained:
	# 	pass
	# 	# logging.info("=> using pre-trained model '{}'".format(args.arch))
	# 	# model = models.__dict__[args.arch](pretrained=True)
	# else:
	logging.info("=> creating model '{}'".format(args.arch))
	model = SpatioTemporalSaliency(CONFIG)
	model._initialize_weights()

	# define loss function (criterion) and optimizer
	criterion = nn.BCELoss().cuda()

#	optimizer = torch.optim.SGD(model.parameters(), args.lr,
#								momentum=args.momentum,
#								weight_decay=args.weight_decay)

	# encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), args.lr,
	# 							betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
	for param in model.encoder.parameters():
		param.requires_grad = False

	optimizer = torch.optim.Adam(model.Custom_CLSTM.parameters(), args.lr,
								betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	cudnn.benchmark = True

	# Data loading code
	config = CONFIG.copy()

	for user in CONFIG['train']:
		config['train'] = [user]
		train_dataset = SequnceDataset(config, 'train')

		if args.resume:
			if os.path.isfile(args.resume):
				logging.info("=> loading checkpoint '{}'".format(args.resume))
				checkpoint = torch.load(args.resume)
				args.start_epoch = checkpoint['epoch']
				best_prec1 = checkpoint['best_prec1']
				model.load_state_dict(checkpoint['state_dict'])
				optimizer.load_state_dict(checkpoint['optimizer'])
				logging.info("=> loaded checkpoint '{}' (epoch {})"
					  .format(args.resume, checkpoint['epoch']))
			else:
				logging.info("=> no checkpoint found at '{}'".format(args.resume))

		train_loader = train_dataset
		# train_loader = torch.utils.data.DataLoader(
		# 	train_dataset, batch_size=args.batch_size, shuffle=False,
		# 	num_workers=args.workers, pin_memory=True, sampler=None)

		# val_loader = torch.utils.data.DataLoader(
		# 	Saliency( CONFIG, 'test'),
		# 	batch_size=args.batch_size, shuffle=False,
		# 	num_workers=args.workers, pin_memory=True)

		# if args.evaluate:
		# 	validate(val_loader, model, criterion)
		# 	return
		
		# if args.visualize:
		# 	visualize(val_loader, model)
		# 	return

		for epoch in range(args.start_epoch, args.epochs):

			adjust_learning_rate(optimizer, epoch)

			# train for one epoch
			train(train_loader, model, criterion, optimizer, epoch, config)

			# evaluate on validation set
			# prec1 = validate(val_loader, model, criterion)

			# remember best prec@1 and save checkpoint
			# is_best = prec1 > best_prec1
			# best_prec1 = max(prec1, best_prec1)
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				# 'best_prec1': best_prec1,
				'optimizer' : optimizer.state_dict(),
			}, is_best)


def train(train_loader, model, criterion, optimizer, epoch, config):
	global output, target_var

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()


	# switch to train mode
	model.train()

	end = time.time()
	for idx, (input, sal, target, img_path) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input).cuda(0)
		target_var = torch.autograd.Variable(target).unsqueeze(1).cuda(1)

		# compute output
		output = model([input_var, sal, target_var, img_path])
		loss = criterion(output[:-1], target_var)

		# measure accuracy and record loss
		# acc  = accuracy(output.data.cpu().numpy(), target.cpu().numpy())
		losses.update(loss.data[0], input.size(0))
		# top1.update(acc.mean(), input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if idx % args.print_freq == 0:
			logging.info('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
				# 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
				epoch, idx, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss=losses))#, top1=top1))


def validate(val_loader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()
	end = time.time()

	correct = list()
	count = list()
	for i, (input, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input, volatile=True).cuda()
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		acc  = accuracy(output.data.cpu().numpy(), target.cpu().numpy())

		losses.update(loss.data[0], input.size(0))
		top1.update(acc.mean(), input.size(0))


		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			logging.info('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
				   i, len(val_loader), batch_time=batch_time, loss=losses,
				   top1=top1))

	logging.info(' * Prec@1 {top1.avg:.3f}'
		  .format(top1=top1))

	return top1.avg


def save_checkpoint(state, is_best, filename='encoder.pth.tar'):
	filename = os.path.join(args.weights, filename)
	torch.save(state, filename)
	if is_best:
		logging.warning('***********************saving best model *********************')
		best = os.path.join(args.weights, 'encoder-best.pth.tar')
		shutil.copyfile(filename, best)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target):
	"""Computes the precision@k for the specified values of k"""
	batch_size = target.shape[0]
	result = list()
	metric = eval(args.metric)

	for i in range(batch_size):
		result.append(metric(output[i], target[i]))

	return np.array(result)

def visualize(loader, model):
	counter = 0
	for batch_idx, (input, target) in enumerate(loader):
		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input, volatile=True).cuda()
		target_var = torch.autograd.Variable(target, volatile=True)
		output = model(input_var).data.cpu().numpy()
		for idx, img in enumerate(input):
			img = Image.open(loader.dataset.dataset[counter][0])
			counter+=1
			w, h = img.size
			mask = np.array(output[idx][0] * 255, dtype=np.uint8)
			mask = Image.fromarray(mask).resize((w,h)).convert('RGB')
			saliency = np.array(target[idx][0] * 255, dtype=np.uint8)
			saliency = Image.fromarray(saliency).resize((w,h)).convert('RGB')
			
			out = Image.new('RGB', (w, h*2))
			out.paste(Image.blend(img, mask, alpha=0.9).convert('RGB'), (0,0))
			out.paste(Image.blend(img, saliency, alpha=0.9).convert('RGB'),(0,h))

			out_path = os.path.join(args.visualize, '{0}-{1}.jpg'.format(batch_idx, idx))
			out.save(out_path)



if __name__ == '__main__':
	main()



# def train():

# 	print('building model')
# 	model = SpatioTemporalSaliency(activation=act)
# 	print('initializing')
# 	model._initialize_weights(True)
# 	print("let's bring on gpus!")
# 	model = model.cuda()


# 	if model.activation == 'softmax':
# 		crit = nn.KLDivLoss()
# 	elif model.activation == 'sigmoid':
# 		crit = nn.MSELoss()
# 	else:
# 		crit = nn.MSELoss()

# 	if opt == 'ADAM':
# 		optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(B1, B2), eps=eps)
# 	elif opt=='SGD':
# 		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


# 	if not fine_tune:
# 		for param in model.encoder.features.parameters():
# 				param.requires_grad = False


# 	print('prep dataset')

# 	d = Salicon(size=size, gamma=gamma)
# 	d.load()

# 	start = time.time()
# 	l = list()
# 	for ep in range(epoch):
# 		total_size = np.array([len(d._map['train'][item]) for item in d._map['train'] ]).sum()
# 		iteration = total_size // batch_size
# 		for step in range(iteration):
# 			if act=='softmax':
# 				batch = d.next_batch(batch_size, mode='train', norm='L1')
# 			else:
# 				batch = d.next_batch(batch_size, mode='train', norm=None)

# 			images = [img[0] for img in batch]
# 			images = torch.stack(images)

# 			seq = np.array([img[1] for img in batch], dtype=np.float32)
# 			seq_input = torch.from_numpy(seq[:,:-1,...]).unsqueeze(2)
# 			#target = torch.from_numpy(seq[:,1:,...]).unsqueeze(2)
# 			target = torch.from_numpy(seq).unsqueeze(2)

# 			images = Variable(images.cuda(), requires_grad=True)
# 			seq_input = Variable(seq_input.cuda(), requires_grad=True)
# 			target = Variable(target.cuda(), requires_grad=False)

# 			#torch.cuda.synchronize()
# 			output = model(images, seq_input)
# 			loss = 0
# 			for t in range(target.size(1)):
# 				loss += crit(output[:,t,...].squeeze(), target[:,t,...].squeeze())

# 			try:
# 				optimizer.zero_grad()
# 				loss.backward()
# 				optimizer.step()
# 				l.append(loss.data[0])
# 				#torch.cuda.synchronize()
# 			except Exception as x:
# 				print(x)
# 				return 0


# 			if step%20 == 0 and step != 0:
# 				print('epoch {0} , step {1} / {2}   , mean-loss: {3}  ,  time: {4}'.format(ep, step, iteration, np.array(l).mean(), time.time()-start))
# 				l = list()
# 				start = time.time()

# 			if step%100 == 0 and step !=0:
# 				try:
# 					#make video
# 					print(step)
# 					print("now generating video!")
# 					video = cv2.VideoWriter()
# 					#success = video.open("{0}/generated_conv_lstm_video_{1}_{2}.avi".format(vid_path, ep, step), fourcc, 4, im_size, True)
# 					writer = skvideo.io.FFmpegWriter("{0}/generated_conv_lstm_video_{1}_{2}.avi".format(vid_path, ep, step))
# 					model.eval()
# 					img = d.next_batch(batch_size=1, mode='test')[0][0]
# 					img_resize = img.resize(im_size)
# 					img_processed = Variable(img_processor(img).unsqueeze(0).cuda())

# 					output = model(img_processed, sequence=None, itr=64)
# 					output = output.permute(0,1,4,3,2)
# 					ims = output[0].squeeze().data.cpu().numpy()
# 					print(ims.shape)
# 					for i in range(ims.shape[0]):
# 						if act=='sigmoid':
# 							x_1_r = np.uint8((np.maximum(ims[i], 0)) * 255)
# 						elif act=='softmax':
# 							x_1_r = np.uint8((np.maximum(ims[i], 0) / ims[i].max()) * 255)
# 						else:
# 							x_1_r = np.uint8(np.minimum(np.maximum(ims[i], 0), 255))
# 							x_1_r = ((x_1_r - x_1_r.min()) / (x_1_r.max() - x_1_r.min())) * 255.0

# 						mask = Image.fromarray(x_1_r).resize(im_size).convert('RGB')
# 						new_im = Image.blend(img_resize, mask, alpha=0.7).convert('RGB')
# 						#new_im = cv2.resize(x_1_r, (256,256))
# 						#video.write(np.asarray(new_im))
# 						writer.writeFrame(new_im)
# 					#video.release()
# 					writer.close()
# 					model.train()

# 				except Exception as x:
# 					print(x)

# 			if step%20000 == 0:
# 				model.save_checkpoint(model.state_dict(), ep, step, path=ck_path)

# 			del images, seq_input, target, loss



# train()

