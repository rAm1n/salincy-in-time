
from dataset import *
from PIL import Image
from tqdm import tqdm
import pickle
import json
import shutil
import torchvision.transforms as transforms
from torch import Tensor
from torch.autograd import Variable
import torch
import glob
from tqdm import tqdm
import threading
from Queue import Queue
import numpy as np
##################################

#pre-processing


d_name = 'CAT2000'
im_size = (224,224)
batch_size = 4
min_len = 5
max_len = 31


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(im_size),
   transforms.ToTensor(),
   normalize
])


def process_seq(seq):


	return out



def build_dataset():
	d = SaliencyBundle(d_name)
	seq = list(d.get('sequence', percentile=True, modify='remove'))
	stim_path = d.get('stimuli_path')


	images = list()
	#########################################
	# removing images with only 1 channel.
	print('starting processs.')
	print('stage 1 - cleaning dataset')
	counter = 0
	for idx, img in enumerate(stim_path):
		img = Image.open(img)
		if img.mode == 'RGB':
			images.append(preprocess(img))
		else:
			counter += 1
			seq.remove(idx)
	images = np.array(images)
	msg = '{0} images were removed out of {1} . current size {2}'
	print( msg.format(counter, len(stim_path), len(images)) )
	###########################################
	#


