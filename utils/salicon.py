
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
	def __init__(self, path='map.pkl', d_name='SALICON', im_size=(224,224), batch_size=4, min_len=50, max_len=550, grid_size=32, gamma=1):

		self.path = path
		self.d_name = d_name
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

		# self._load_data()
		# self._preprocess()

		self.preprocessed = False
		self.images = list()
		self.sequences = list()
		self._map = {i:list() for i in range(self.min_len, self.max_len + 1)}

	def initialize(self):
		if not self.preprocessed:
			self._load_data()
			self.preprocess()
			self.preprocessed = True
		else:
			# just to rebuild main, won't take long
			self.reload_map()


	def reload_map(self, path):
		if not path:
			path = self.path
		self._map = pickle.load(path)


	def _load_data(self):
		print('stage 1 - start loading data.')
		self.d = SaliencyBundle(self.d_name)
		self.raw_seq = list(self.d.get('sequence', percentile=True, modify='remove'))
		self.stim_path = self.d.get('stimuli_path')


	def _preprocess(self):
		print('stage 2 - cleaning dataset - preprocess')
		for img_idx, img in enumerate(self.stim_path):
			img = Image.open(img)
			if img.mode == 'RGB':
				p = self.img_preprocess(img)
				# threshold on seq length
				# tmp = list()
				for s in self.raw_seq[img_idx]:
					shape = s.shape
					if (shape[0] >= self.min_len) and (shape[0] <= self.max_len):
						# z = np.zeros((shape[0], self.grid_size, self.grid_size))
						# s = s[:,:2]
						# for idx, row in enumerate(s):
						# 	h = int(self.grid_size * row[0])
						# 	w = int(self.grid_size * row[1])
							# z[idx][h][w] = 1
							# z[idx] = gaussian_filter(z[idx], self.gamma)
						# z = z.astype(np.float16)
						self.sequence.append(s)
						seq_idx = len(self.sequence) - 1
						self._map[shape[0]].append((img_idx,seq_idx))
				self.images.append(p)

		print('stage 3 - shuffling samples')
		for key in self.main:
			random.shuffle(self.main[key])




	def next_batch(self):
		batch = list()
		while True:
			if not self.main.keys():
				self.initialize()
			random_len = random.choice(self.main.keys())
			if len(main[random_len]) > batch_size:
				break
			else:
				del main[random_len]

		for i in range(self.batch_size):
			img_idx , seq_idx = main[random_len].pop()
			batch.append([self.images[img_idx], self.sequence[seq_idx]])

		return batch
