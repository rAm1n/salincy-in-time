
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
	def __init__(self, path='map.pkl', d_name='SALICON', im_size=(224,224), batch_size=2, min_len=50, max_len=550, grid_size=32, gamma=1):

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

		normalize = transforms.Normalize(
		   mean=[0.485, 0.456, 0.406],
		   std=[0.229, 0.224, 0.225]
		)

		self.img_preprocess = transforms.Compose([
		   transforms.Scale(im_size),
		   transforms.ToTensor(),
		   normalize
		])

		self.images = list()
		self.sequences = list()
		self._map = {i:list() for i in range(self.min_len, self.max_len + 1)}




	def initialize(self):
		self._load_data()
		self._preprocess()


	def reload_map(self, path=None):
		if not path:
			path = self.path
		with open(path, 'r') as handle:
			self._map = pickle.load(handle)


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
				for s in self.raw_seq[img_idx]:
					shape = s.shape
					if (shape[0] >= self.min_len) and (shape[0] <= self.max_len):
						self.sequences.append(s)
						seq_idx = len(self.sequences) - 1
						self._map[shape[0]].append((img_idx,seq_idx))
				self.images.append(p)

		print('stage 3 - shuffling samples')
		for key in self._map:
			random.shuffle(self._map[key])

		print('stage 4 - saving map')
		with open(self.path, 'w') as handle:
			pickle.dump(self._map, handle)


	def next_batch(self):
		batch = list()
		while True:
			if not self._map.keys():
				print('next epoch')
				self.reload_map()
			random_len = random.choice(self._map.keys())
			if len(self._map[random_len]) >  self.batch_size:
				break
			else:
				del self._map[random_len]

		for i in range(self.batch_size):
			img_idx , seq_idx = self._map[random_len].pop()
			seq = self.sequences[seq_idx][:,:2]
			shape = seq.shape
			z = np.zeros((int(shape[0]/4) + 1 , self.grid_size, self.grid_size))
			for idx, row in enumerate(seq):
				if (idx%4) == 0:
					idx/=4
					h = int(self.grid_size * row[0])
					w = int(self.grid_size * row[1])
					z[idx][h][w] = 1
					z[idx] = gaussian_filter(z[idx], self.gamma)
			z = z.astype(np.float16)
			batch.append([self.images[img_idx], z])

		return batch

