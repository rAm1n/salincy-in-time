
from dataset import *
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import hickle as pickle
from scipy.ndimage.filters import gaussian_filter
from random import shuffle
import os

class Salicon():
	def __init__(self, path='tmp/', d_name='SALICON', im_size=(224,224), min_len=50, max_len=550, seq_len= 16, grid_size=32, gamma=1, max_thread=8):

		self.path = path
		self.d_name = d_name
		self.im_size = im_size
		self.min_len = min_len
		self.max_len = max_len
		self.seq_len = seq_len
		self.grid_size = grid_size
		self.gamma = gamma
		self.max_thread= max_thread

		self.sizes = {'train': [0,0.9], 'validation': [.9, 0.95], 'test': [.95, 1]}

		self.index = {'train':0, 'validation':0, 'test': 0}


		self.img_processor = transforms.Compose([
		   transforms.Scale(im_size),
		   transforms.ToTensor(),
		   transforms.Normalize(
		   mean=[0.485, 0.456, 0.406],
		   std=[0.229, 0.224, 0.225]
			)
		])

		self.images = dict({key:list() for key in ['train','validation','test']})
		self.sequences = dict({key:list() for key in ['train','validation','test']})
		self._map = dict({key:list() for key in ['train','validation','test']})




	def initialize(self):
		self._load_data()
		self._preprocess()

	def load(self):
		# check if files exists
		path = os.path.join(self.path, 'map.pkl')
		if os.path.exists(path):
			with open(path, 'r') as handle:
				self._map = pickle.load(handle)

			path = os.path.join(self.path, 'images.pkl')
			if os.path.exists( path):
				with open(path, 'r') as handle:
					self.images = pickle.load(handle)

                        path = os.path.join(self.path, 'sequences.pkl')
                        if os.path.exists( path):
                                with open(path, 'r') as handle:
                                        self.sequences = pickle.load(handle)

		else:
			self.initialize()


	def _load_data(self):
		print('start loading data.')
		self.dataset = SaliencyBundle(self.d_name)
		raw_seq = self.dataset.get('sequence', percentile=True, modify='remove')[:5000]
		stim_path = self.dataset.get('stimuli_path')[:5000]

		total_size = len(stim_path)

		print('spliting data')
		self.raw_seq = dict({key:list() for key in ['train','validation','test']})
		self.stim_path = dict({key:list() for key in ['train','validation','test']})

		for key in self.sizes:
			index = (int(self.sizes[key][0] * total_size), int(self.sizes[key][1] * total_size))
			self.raw_seq[key] = raw_seq[ index[0] : index[1]]
			self.stim_path[key] = stim_path[ index[0] : index[1]]


	def _preprocess(self):
		print('preprocessing - takes a damn while, be patient...')
		for key in self.stim_path:
			# choosing set -> train, validation, test
			dataset = self.stim_path[key]
			for img_idx, img in enumerate(dataset):
				img = Image.open(img)
				if img.mode == 'RGB':
					img_processed = self.img_processor(img)
					self.images[key].append(img_processed)
					for seq in self.raw_seq[key][img_idx]:
						shape = seq.shape
						if (shape[0] >= self.min_len) and (shape[0] <= self.max_len):
							mini_seq = list()
							old_fix = (0,0)
							for fix in seq:
								h = int(self.grid_size * fix[0])
								w = int(self.grid_size * fix[1])
								fix = (h,w)
								if fix != old_fix:
									mini_seq.append(fix)
									old_fix = fix
									if len(mini_seq) == self.seq_len:
										self.sequences[key].append(np.array(mini_seq, dtype=np.int16))
										self._map[key].append((img_idx, len(self.sequences[key]) -1 ))
										mini_seq = list()

			shuffle(self._map[key])

		del self.raw_seq
		del self.stim_path
#		print('Saving map ...')
#		with open(os.path.join(self.path, 'map.pkl'), 'w') as handle:
#			pickle.dump(self._map, handle, compression='gzip')
#		with open(os.path.join(self.path, 'images.pkl'), 'w') as handle:
#			pickle.dump(self.images, handle, compression='gzip')
#                with open(os.path.join(self.path, 'sequences.pkl'), 'w') as handle:
#                       pickle.dump(self.sequences, handle, compression='gzip')



	def next_batch(self, batch_size=2, mode='train'):
		batch = list()
		try:
			for i in range(batch_size):
				index = self.index[mode]
				img_idx , seq_idx = self._map[mode][index]
				raw_seq = self.sequences[mode][seq_idx][:,:2]


				seq = list() #processed

				for idx, fix in enumerate(raw_seq):
						z = np.random.uniform(low=0, high=0.005, size=( self.grid_size, self.grid_size))
						z[fix[0]][fix[1]] = 2
						z = gaussian_filter(z, self.gamma)
						seq.append(z / z.sum())

				batch.append([self.images[mode][img_idx], np.array(seq, dtype=np.float16)])

				# updating index
				self.index[mode] += 1
				if self.index[mode] >= len(self.images[mode]):
					self.index[mode] = 0

		except Exception,x:
			print x

		return batch

