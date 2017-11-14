
from dataset import *
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import pickle
from scipy.ndimage.filters import gaussian_filter
import random


class Salicon():
	def __init__(self, path='', d_name='SALICON', im_size=(224,224), batch_size=4, min_len=5, max_len=30, grid_size=224, gamma=1):

		self.path = path
		self.d_name = 'SALICON'
		self.im_size = im_size
		# self.set_size = set_size
		self.batch_size = batch_size
		self.min_len = min_len
		self.max_len = max_len
		self.grid_size = grid_size
		self.gamma = gamma
		self.pointer = 0

		# self.sets = [list(), list(), list()] #train - validation - test

		normalize = transforms.Normalize(
		   mean=[0.485, 0.456, 0.406],
		   std=[0.229, 0.224, 0.225]
		)
		self.img_preprocess = transforms.Compose([
		   transforms.Scale(im_size),
		   transforms.ToTensor(),
		   normalize
		])

		self._load_data()
		self._preprocess()

	def _load_data(self)
		print('stage 1 - start loading data.')
		self.d = SaliencyBundle(d_name)
		self.raw_seq = list(self.d.get('sequence', percentile=True, modify='remove'))
		self.stim_path = self.d.get('stimuli_path')


	def _preprocess(self):
		self.images = list()
		self.sequence = list()
		self.main = main = {i:list() for i in range(min_len, max_len + 1)}
		#########################################
		# removing images with only 1 channel.
		print('stage 2 - cleaning dataset - preprocess')
		for img_idx, img in enumerate(self.stim_path):
			img = Image.open(img)
			if img.mode == 'RGB':
				p = self.img_preprocess(img)
				# threshold on seq length
				tmp = list()
				for s in self.raw_seq[img_idx]:
					shape = s.shape
					if (shape[0] >= self.min_len) and (shape[0] <= self.max_len):
						z = np.zeros((shape[0], self.grid_size, self.grid_size), dtype=np.float16)
						s = s[:,:2]
						for idx, row in enumerate(s):
							h = int(grid_size * row[0])
							w = int(grid_size * row[1])
							z[idx][h][w] = 1
							z[idx] = gaussian_filter(z[idx], gamma)
						main[shape[0]].append((img_idx,z))

				self.images.append(p)

		for key in main:
			random.shuffle(main[key])



	def next_batch(self):
		batch = list()
		for i in range(self.batch_size):
			random_len = random.randrange(self.min_len, self.max_len + 1)
			img_idx , z = main[key].pop()
			batch.append([self.images[img_idx], z])

		return batch
