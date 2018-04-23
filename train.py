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
import skvideo.io
import skimage.transform
import math
import pickle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model import RNNSaliency
from dataset import SequnceDataset
from config import CONFIG, MODELS
from utils import extract_model_fixations
from saliency.metrics import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Scanpath prediction')

parser.add_argument('--mode', '-m', metavar='MODE', default='train',
 					choices=['train', 'eval'],
 					help='evaluation metric')
# parser.add_argument('--weights', default='/media/ramin/data/scanpath/weights/', metavar='DIR',
# 					help='path to weights')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='DVGG16CLSTM1',
# 					choices=['DVGG16CLSTM1', 'DVGG16CLSTM2', 'DVGG16BCLSTM3'],
# 					help='enocer architecture: ' +
# 						' (default: DVGG16CLSTM1)')
parser.add_argument('--log', default='logs/', metavar='DIR',
 					help='path to dataset')
# parser.add_argument('--metric', metavar='METRIC', default='AUC',
# 					choices=['MultiMatch', 'DTW', 'frechet_distance',
# 					'euclidean_distance','hausdorff_distance', 'levenshtein_distance'],
# 					help='evaluation metric')
# parser.add_argument('-v','--visualize', default='/media/ramin/data/scanpath/visualization/', metavar='DIR',
# 					help='path to dataset')
# parser.add_argument('--visualize-count', default=50, type=int, metavar='N',
# 					help='number of images to visualize')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
# 					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
 					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
 					help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#  					metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
# 					metavar='LR', help='initial learning rate')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
# 					help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
# 					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
 					metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
# 					help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
# 					help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
# 					help='use pre-trained model')

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cudnn.benchmark = True

best_prec1 = 0


if 'MultiMatch' in CONFIG['eval']['metrics']:
	matlab = make_engine()
else:
	matlab = None


def main():
	global args, best_prec1, model, train_dataset, val_dataset

	args = parser.parse_args()


#
#	# Data loading code
	config = CONFIG.copy()

	mode = args.mode

	for model_name in MODELS:
		# setting up the logger.
		log = logging.getLogger()
		for hdlr in log.handlers[:]:
			if isinstance(hdlr,logging.FileHandler):
				log.removeHandler(hdlr)

		logging.basicConfig(
				format="%(message)s",
				handlers=[
					logging.FileHandler("{0}/{1}-{2}.log".format(
						args.log, model_name ,sys.argv[0].replace('.py','') + datetime.now().strftime('_%H_%M_%d_%m_%Y'))),
					logging.StreamHandler()
				],
				level=logging.INFO)

		for user in CONFIG[mode]['users']:
			config[mode]['users'] = [user]
			config['model']['name'] = model_name

			train_dataset = SequnceDataset(config, mode)

			#logging.info("=> creating model'{}'".format(args.arch))
			if config['model']['type'] == 'RNN':
				model = RNNSaliency(config)
			else:
				pass

			model._initialize_weights()

			for param in model.encoder.parameters():
				param.requires_grad = False

			criterion = nn.BCELoss().cuda()

			optimizer = torch.optim.Adam(model.decoder.parameters(), config['train']['lr'],
										betas=(0.9, 0.999), eps=1e-08, weight_decay=config['train']['weight_decay'])

			# if args.resume:
			# 	if os.path.isfile(args.resume):
			# 		logging.info("=> loading checkpoint '{}'".format(args.resume))
			# 		checkpoint = torch.load(args.resume)
			# 		args.start_epoch = checkpoint['epoch']
			# 		# best_prec1 = checkpoint['best_prec1']
			# 		model.load_state_dict(checkpoint['state_dict'])
			# 		optimizer.load_state_dict(checkpoint['optimizer'])
			# 		logging.info("=> loaded checkpoint '{}' (epoch {})"
			# 			  .format(args.resume, checkpoint['epoch']))
			# 	else:
			# 		logging.info("=> no checkpoint found at '{}'".format(args.resume))

			# train_loader = train_dataset
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
			# 	visualize(train_loader, model, user, epoch)
			# 	return

			for epoch in range(args.start_epoch, args.epochs):

				adjust_learning_rate(optimizer, epoch)

				# train for one epoch
				train(train_dataset, model, criterion, optimizer, epoch, config)

				# visualize(train_loader, model, str(user), str(epoch))
				# evaluate on validation set
				validate(train_dataset, model, criterion, user, epoch, config)


				# remember best prec@1 and save checkpoint
				# is_best = prec1 > best_prec1

				# best_prec1 = max(prec1, best_prec1)

				save_checkpoint({
					'epoch': epoch + 1,
					'user': user,
					'arch': config['model']['name'],
					'state_dict': model.state_dict(),
					# 'best_prec1': best_prec1,
					'optimizer' : optimizer.state_dict(),
				})


def train(train_loader, model, criterion, optimizer, epoch, config):
	global output, target_var

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	prec = list()

	for metric in config['train']['metrics']:
		prec.append(AverageMeter())


	# switch to train mode
	model.train()

	end = time.time()

	for idx, x in enumerate(train_loader):
		# measure data loading time
		if x is None:  # skip samples removed with preprocessing policy
			continue

		t,c,h,w = x['input'].size()

		data_time.update(time.time() - end)

		target = x['gts'].cuda(async=True)
		history = x['history'].cuda(async=True)

		input_var = torch.autograd.Variable(x['input'], volatile=True).cuda()
		target_var = torch.autograd.Variable(target, volatile=True)
		history_var = torch.autograd.Variable(history, volatile=True)

		# compute output
		landa = self.config['train']['landa']
		output = model([input_var, x['saliency'], target_var, x['img_path']])
		loss = 0.0
		for t in range(target_var.size(0)):
			loss += ( criterion(output[t][0], target_var[t]) +\
				 landa * (history * output[t][0]).sum() )
		# loss = criterion(output[:-1], target_var)

		losses.update(loss.data[0], x['input'].size(0))
		# measure accuracy and record loss

		output = output.data.cpu().numpy().squeeze()
		output_fixations = extract_model_fixations(output, (h,w))

		for m_idx, metric in enumerate(config['train']['metrics']):
			acc  = accuracy(output_fixations, x['fixations'], metric)
			prec[m_idx].update(acc, x['input'].size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if idx % args.print_freq == 0:
				msg = 'TRAINING - User/Epoch: [{0}][{1}][{2}/{3}]\t' \
				'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t' \
				'Loss {loss_val:.4f} ({loss_avg:.4f})\n\n'.format(
				config['train']['users'][0], epoch, idx, len(train_loader),
				batch_time_val=batch_time.val[0], batch_time_avg=batch_time.avg[0],
				loss_val=losses.val[0], loss_avg=losses.avg[0])

				# msg = 'User/Epoch: [{0}][{1}][{2}/{3}]\t' \
				# 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
				# 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
				# 'Loss {loss.val:.4f} ({loss.avg:.4f})\n\n'.format(
				# config['train']['users'][0], epoch, idx, len(train_loader), batch_time=batch_time,
				# data_time=data_time, loss=losses)

				for m_idx, metric in enumerate(config['train']['metrics']):
					val = ' '.join([str(round(num,3)) for num in prec[m_idx].val.flatten()])
					avg = ' '.join([str(round(num,3)) for num in prec[m_idx].avg.flatten()])
					msg += '{0}\t{1}\t({2})\n'.format(metric, val, avg)

				logging.info(msg)


def validate(val_loader, model, criterion, user, epoch, config):
	batch_time = AverageMeter()
	losses = AverageMeter()
	prec = list()


	# saving everything in pickle.
	voloums = list()
	fixations = list()
	evaluations = list()



	for metric in config['eval']['metrics']:
		prec.append(AverageMeter())

	# switch to evaluate mode
	model.eval()
	end = time.time()


	for idx, x in enumerate(val_loader):

		if x is None:  # skip samples removed with preprocessing policy
			fixations.append(None)
			voloums.append(None)
			evaluations.append({ metric: None for metric in config['eval']['metrics']})
			continue


		b,t, h,w = x['input'].size()

		target = x['gts'].cuda(async=True)
		input_var = torch.autograd.Variable(x['input'], volatile=True).cuda()
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		#output = model(input_var)

		output = model([input_var, x['saliency'], target_var, x['img_path']])

		# measure accuracy and record loss
		loss = 0.0
		for t in range(target_var.size(0)):
			loss += criterion(output[t][0], target_var[t])

		losses.update(loss.data[0], x['input'].size(0))


		output = output.data.cpu().numpy().squeeze()
		output_fixations = extract_model_fixations(output, (h,w))
		voloums.append(output)
		fixations.append(output_fixations)


		tmp = dict()
		for m_idx, metric in enumerate(config['eval']['metrics']):
			acc  = accuracy(output_fixations, x['fixations'], metric, height=h, width=w, matlab_engine=matlab)
			prec[m_idx].update(acc)
			tmp[metric] = prec[m_idx].val.flatten()
		evaluations.append(tmp)


		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		if idx % args.print_freq == 0:
				msg = 'VAL - User/Epoch: [{0}][{1}][{2}/{3}]\t' \
				'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t' \
				'Loss {loss_val:.4f} ({loss_avg:.4f})\n\n'.format(
				config['train']['users'][0], epoch, idx, len(val_loader),
				batch_time_val=batch_time.val[0], batch_time_avg=batch_time.avg[0],
				loss_val=losses.val[0], loss_avg=losses.avg[0])

				for m_idx, metric in enumerate(config['eval']['metrics']):
					val = ' '.join([str(round(num,3)) for num in prec[m_idx].val.flatten()])
					avg = ' '.join([str(round(num,3)) for num in prec[m_idx].avg.flatten()])
					msg += '{0}\t{1}\t({2})\n'.format(metric, val, avg)

				logging.info(msg)


	filename = '{}-{}-{}.pkl'
	filename = os.path.join(config['eval_path'], filename.format(CONFIG['model']['name'], user, epoch))

	with open(filename,'wb') as handle:
		tmp = {}
		tmp['config'] = config
		tmp['fixations'] = np.array(fixations)
		tmp['voloums'] = np.array(voloums)
		tmp['eval'] = evaluations

		pickle.dump(tmp, handle)

	# return top1.avg


def save_checkpoint(state, filename='{0}_U{1}_E{2}.pth.tar'):

	path = os.path.join(CONFIG['weights_path'], filename.format(CONFIG['model']['name'],state['user'], state['epoch']))
	# if is_best:
	# 	logging.warning('***********************saving best model *********************')
	# 	best = os.path.join(args.weights, 'encoder-best.pth.tar')
	# 	shutil.copyfile(filename, best)
	torch.save(state, path)



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
		if not isinstance(val, np.ndarray):
			val = np.array([val])

		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = CONFIG['train']['lr'] * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, metric, **kwargs):
	"""Computes the precision@k for the specified values of k"""

	return eval(metric)(P=output, Q=target, **kwargs)




# def visualize(loader, model, user, epoch):
# 	counter = 0
# 	model.eval()

# 	path = os.path.join(CONFIG['visualize'], CONFIG['model']['name'], user, epoch)
# 	if not os.path.exists(path):
# 		os.makedirs(path)

# 	for idx, (input, sal, target, img_path) in enumerate(loader):
# 		# measure data loading time
# 		counter+=1
# 		if counter > args.visualize_count:
# 			break

# 		target = target.cuda(async=True)
# 		input_var = torch.autograd.Variable(input).cuda(0)
# 		target_var = torch.autograd.Variable(target).unsqueeze(1).cuda()
# 		output = model([input_var, sal, target_var, img_path]).data.cpu().numpy()
# 		img = Image.open(img_path)
# 		w, h = img.size

# 		len_out = output.shape[0]

# 		for seq_idx, tar in enumerate(target):
# 			if seq_idx >= len_out:
# 				break
# 			mask = np.array(output[seq_idx][0] * 255, dtype=np.uint8)
# 			mask = Image.fromarray(mask).resize((w,h)).convert('RGB')
# 			saliency = np.array(tar * 255, dtype=np.uint8)
# 			saliency = Image.fromarray(saliency).resize((w,h)).convert('RGB')

# 			out = Image.new('RGB', (w, h*2))
# 			out.paste(Image.blend(img, mask, alpha=0.7).convert('RGB'), (0,0))
# 			out.paste(Image.blend(img, saliency, alpha=0.7).convert('RGB'),(0,h))

# 			out_path = os.path.join(path, '{0}-{1}.jpg'.format(idx, seq_idx))
# 			out.save(out_path)

# def visualize_image(img_path, output, config, user, epoch):

# 	path = os.path.join(CONFIG['visualize'], CONFIG['model']['name'], user, epoch)
# 	if not os.path.exists(path):
# 		os.makedirs(path)

# 	for mask in output:

# 		if self.config['eval']['next_frame_policy']=='max':
# 			x_max, y_max = np.unravel_index(mask.argmax(), mask.shape)

# 			mask, _ = fov_mask(img.size[::-1], radius=self.config['dataset']['fixation_radius'],
# 							center=(x_mask,y_mask), th=self.config['eval']['mask_th'])




if __name__ == '__main__':
	main()
