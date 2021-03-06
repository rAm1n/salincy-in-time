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
import glob


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model import SpatioTemporalSaliency
from dataset import SequnceDataset
from config import CONFIG


parser = argparse.ArgumentParser(description='Scanpath prediction')

parser.add_argument('--weights', default='/media/ramin/data/scanpath/weights-final/', metavar='DIR',
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

parser.add_argument('-v','--visualize', default='/media/ramin/data/scanpath/viz-single/', metavar='DIR',
					help='path to dataset')

parser.add_argument('--visualize-count', default=50, type=int, metavar='N',
					help='number of images to visualize')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
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
	# logging.info("=> creating model '{}'".format(args.arch))

	model = SpatioTemporalSaliency(CONFIG)
	model._initialize_weights()
	model.eval()


	# define loss function (criterion) and optimizer
	criterion = nn.BCELoss().cuda()

	for param in model.parameters():
		param.requires_grad = False

	# optionally resume from a checkpoint
	cudnn.benchmark = True

	# Data loading code
	config = CONFIG.copy()

	epochs = args.epochs
	users = 15
	len_dataset = 700
	masks = np.zeros((epochs, users, len_dataset, 8, 75, 100), dtype=np.float16)

	bad_list = set([231,310,372,317,436,441,447,541,535,675,661])

	for epoch in range(1,epochs+1):
		print('starting forward pass for epoch {0}'.format(epoch))
		for user in CONFIG['test']:
			print('starting user {0}'.format(user+1))
			config['test'] = [user]
			test_dataset = SequnceDataset(config, 'train')

			# Let's resume weights.
			w_path = os.path.join(args.weights, '{0}_{1}.pth.tar'.format(user, epoch))
			if os.path.isfile(w_path):
				logging.info("=> loading checkpoint '{}'".format(args.resume))
				checkpoint = torch.load(w_path)
				args.start_epoch = checkpoint['epoch']
				model.load_state_dict(checkpoint['state_dict'])
				logging.info("=> loaded checkpoint '{}' (epoch {})"
					  .format(args.resume, checkpoint['epoch']))
			else:
				logging.info("=> no checkpoint found at '{}'".format(args.resume))

			start = time.time()
			for img_idx, (input, sal, target, img_path) in enumerate(test_dataset):
				img_idx = int(img_path.split('/')[-1].split('.jpg')[0]) - 1001
				print(epoch, user, img_idx, len(test_dataset))
				if img_idx in bad_list:
					continue
				# measure data loading time
				input_var = torch.autograd.Variable(input, volatile=True).cuda(0)
				output = model([input_var, sal, target, img_path])
				masks[epoch-1][user][img_idx] = output[0,0]

				img = Image.open(img_path)
				w, h = img.size

				len_out = output.shape[0]

				counter = 0
				path = os.path.join(args.visualize, str(user), str(epoch))
				if not os.path.exists(path):
					os.makedirs(path)

				for seq_idx, tar in enumerate(output):
					if seq_idx >= len_out:
						break
					mask = np.array(output[seq_idx][0] * 255, dtype=np.uint8)
					mask = Image.fromarray(mask).resize((w,h)).convert('RGB')
					saliency = np.array(tar * 255, dtype=np.uint8)
					saliency = Image.fromarray(saliency).resize((w,h)).convert('RGB')

					out = Image.new('RGB', (w, h*2))
					out.paste(Image.blend(img, mask, alpha=0.7).convert('RGB'), (0,0))
					out.paste(Image.blend(img, saliency, alpha=0.7).convert('RGB'),(0,h))

					out_path = os.path.join(path, '{0}-{1}.jpg'.format(img_idx, seq_idx))
					out.save(out_path)

			print(time.time()-start)


if __name__ == '__main__':
	main()
