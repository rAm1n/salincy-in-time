import argparse
import os
import shutil
import time
import numpy as np
import logging
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets



from encoder import config as en_config
from encoder import make_encoder
from dataset import Saliency
from config import CONFIG

from metrics import *


parser = argparse.ArgumentParser(description='saliency - training script')

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
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')



best_prec1 = 0


def main():
	global args, best_prec1, model, train_dataset, val_loader
	args = parser.parse_args()


	logging.basicConfig(
		format="%(message)s",
		handlers=[
			logging.FileHandler("{0}/{1}.log".format(args.log, sys.argv[0].replace('.py','') + datetime.now().strftime('_%H_%M_%d_%m_%Y'))),
			logging.StreamHandler()
		],
		level=logging.INFO)

	# create model
	if args.pretrained:
		pass
		# logging.info("=> using pre-trained model '{}'".format(args.arch))
		# model = models.__dict__[args.arch](pretrained=True)
	else:
		logging.info("=> creating model '{}'".format(args.arch))
		model = make_encoder(config=en_config[args.arch])




	model.cuda()

	# define loss function (criterion) and optimizer
	criterion = nn.BCELoss().cuda()

#	optimizer = torch.optim.SGD(model.parameters(), args.lr,
#								momentum=args.momentum,
#								weight_decay=args.weight_decay)

	optimizer = torch.optim.Adam(model.parameters(), args.lr,
								betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	cudnn.benchmark = True

	# Data loading code

	train_dataset = Saliency(CONFIG, 'train')


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

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True, sampler=None)

	val_loader = torch.utils.data.DataLoader(
		Saliency( CONFIG, 'test'),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	if args.evaluate:
		validate(val_loader, model, criterion)
		return

	for epoch in range(args.start_epoch, args.epochs):

		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		# evaluate on validation set
		prec1 = validate(val_loader, model, criterion)

		# remember best prec@1 and save checkpoint
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer' : optimizer.state_dict(),
		}, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()


	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input).cuda()
		target_var = torch.autograd.Variable(target)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		acc  = accuracy(output.data.cpu().numpy(), target.cpu().numpy())
		losses.update(loss.data[0], input.size(0))
		top1.update(acc.mean(), input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			logging.info('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1))


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



if __name__ == '__main__':
	main()
