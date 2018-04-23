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
import pickle


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model import SpatioTemporalSaliency
from dataset import SequnceDataset
from config import CONFIG
from saliency.dataset import SaliencyDataset
from utils import fov_mask


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

	pkls = glob.glob('/media/ramin/data/scanpath/eval/*/*.pkl')
	out_path = '/media/ramin/data/scanpath/visualization-2/'

	d = SaliencyDataset('OSIE')
	imgs = d.get('stimuli_path')


	for pkl in pkls:

		data = pickle.load(open(pkl,'rb'))
		pkl = pkl.replace('.pkl','')
		policy = pkl.split('/')[-2]
		model_name, depth, user, epoch = pkl.split('/')[-1].split('-')
		sequence = d.get('sequence')[:,int(user)]

		path = os.path.join(out_path, model_name, policy, user, epoch)
		print(path)

		if not os.path.exists(path):
			os.makedirs(path)

		for idx, img in enumerate(imgs):
			print(idx)

			if idx > 50:
				break

			img = Image.open(img)
			w, h= img.size

			volume = data['voloums'][idx]
			if volume is not None:
				pass


			for seq_idx, seq in enumerate(sequence[idx]):
				try:
					# mask = volume[seq_idx] /  volume[seq_idx].max()
					mask = volume[seq_idx]
					mask = np.array(mask * 1000, dtype=np.uint8)
					mask = Image.fromarray(mask).resize((w,h)).convert('RGB')


					_, saliency = fov_mask(size=(h,w), center=(seq[:2]))
					saliency = np.array(saliency * 255, dtype=np.uint8)
					saliency = Image.fromarray(saliency).resize((w,h)).convert('RGB')

					out = Image.new('RGB', (w, h*2))
					out.paste(Image.blend(img, mask, alpha=0.7).convert('RGB'), (0,0))
					out.paste(Image.blend(img, saliency, alpha=0.7).convert('RGB'),(0,h))

					image_out_path = os.path.join(path, '{0}-{1}.jpg'.format(idx, seq_idx))
					out.save(image_out_path)
				except Exception as e:
					pass




if __name__ == '__main__':
	main()