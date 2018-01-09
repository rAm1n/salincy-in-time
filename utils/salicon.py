
from .dataset import *
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import pickle
from scipy.ndimage.filters import gaussian_filter
from random import shuffle
import os
from scipy.spatial import distance
from scipy.stats import entropy
import random


class Salicon():
	def __init__(self, path='tmp/', d_name='SALICON', size=10000, im_size=(256,256), min_seq= 5, max_seq=500, min_dist=2.5, max_dist=20, grid_size=32, sigma=2):

		self.path = path
		self.d_name = d_name
		self.im_size = im_size
		self.grid_size = grid_size
		self.size = size
		self.sigma = sigma
		self.min_dist = min_dist
		self.max_dist = max_dist
		self.min_seq = min_seq
		self.max_seq = max_seq

		self.sizes = {'train': [0,0.9], 'validation': [.9, 0.95], 'test': [.95, 1]}

		self.img_processor = transforms.Compose([
		   transforms.Resize(im_size),
		   transforms.ToTensor(),
		   transforms.Normalize(
		   mean=[0.485, 0.456, 0.406],
		   std=[0.229, 0.224, 0.225]
			)
		])

		self.images = dict({key:list() for key in ['train','validation','test']})
		self.sequences = dict({key:list() for key in ['train','validation','test']})
		self._map = dict({key:list() for key in ['train','validation','test']})
		self.index = dict()

		for key in self._map:
			self._map[key] = {i:list() for i in range(min_seq, max_seq + 1)}
			self.index[key] = {i: 0 for i in range(min_seq, max_seq + 1)} 



	def initialize(self):
		self._load_data()
		#self._preprocess()
		self._preprocess_single()


	def load(self):
		# check if files exists
		path = os.path.join(self.path, 'dataset-{0}.pkl'.format(self.size))
		if os.path.exists(path):
			with open(path, 'r') as handle:
				tmp = pickle.load(handle)
				self._map = tmp['map']
				self.images = tmp['images']
				self.sequences = tmp['sequences']
				self.index = tmp['index']
				self.stim_path = tmp['stim_path']
				self.saliency_path = tmp['saliency_path']
		else:
			self.initialize()


	def _load_data(self):
		print('start loading data.')
		self.dataset = SaliencyBundle(self.d_name)
		raw_seq = self.dataset.get('sequence', percentile=True, modify='remove')[:self.size]
		stim_path = self.dataset.get('stimuli_path')[:self.size]
		saliency_path = self.dataset.get('heatmap_path')[:self.size]

		total_size = len(stim_path)

		print('spliting data')
		self.raw_seq = dict({key:list() for key in ['train','validation','test']})
		self.stim_path = dict({key:list() for key in ['train','validation','test']})
		self.saliency_path = dict({key:list() for key in ['train','validation','test']})
		
		for key in self.sizes:
			index = (int(self.sizes[key][0] * total_size), int(self.sizes[key][1] * total_size))
			self.raw_seq[key] = raw_seq[ index[0] : index[1]]
			self.stim_path[key] = stim_path[ index[0] : index[1]]
			self.saliency_path[key] = saliency_path[ index[0] : index[1]]


	def _preprocess(self):
		print('preprocessing - takes a while, be patient...')
		for key in self.stim_path:
			# choosing set -> train, validation, test
			dataset = self.stim_path[key]
			for img_idx, img in enumerate(dataset):
				self.images[key].append(None)
				for seq in self.raw_seq[key][img_idx]:
					shape = seq.shape
					flag = False
					mini_seq = list()
					old_fix = (int(self.grid_size * seq[0][0]), int(self.grid_size * seq[0][1]))
					for fix in seq:
						h = int(self.grid_size * fix[0])
						w = int(self.grid_size * fix[1])
						fix = (h,w)
						dist = self._euc(fix, old_fix)
						if dist > self.max_dist:
							flag = True
							break
						elif dist > self.min_dist:
							mini_seq.append(fix)
							old_fix = fix
							# if len(mini_seq) == self.seq_len:
							# 	self.sequences[key].append(np.array(mini_seq, dtype=np.int16))
							# 	self._map[key].append((img_idx, len(self.sequences[key]) -1 ))
							# 	mini_seq = list()
					if not flag:
						l = len(mini_seq)
						if (l <= self.max_seq) and (l >= self.min_seq):
							self.sequences[key].append(np.array(mini_seq, dtype=np.int16))
							self._map[key][l].append((img_idx, len(self.sequences[key]) -1 ))
			#img.close()
			for l in self._map[key]:
				shuffle(self._map[key][l])

		del self.raw_seq

		print('Saving map ...')
		with open(os.path.join(self.path, 'dataset-{0}.pkl'.format(self.size)), 'w') as handle:
			out = dict()
			out['images'] = self.images
			out['sequences'] = self.sequences
			out['stim_path'] = self.stim_path
			out['saliency_path'] = self.saliency_path
			out['map'] = self._map
			out['index'] = self.index
			pickle.dump(out, handle)#, compression='gzip')


	def _preprocess_single(self):
		for key in self.stim_path:
            # choosing set -> train, validation, test
			dataset = self.stim_path[key]
			for img_idx, img in enumerate(dataset):
				self.images[key].append(None)
				saliency = Image.open(self.saliency_path[key][img_idx])
				saliency = np.array(saliency.resize((self.grid_size, self.grid_size)).getdata(), dtype=np.float16)
				#saliency = saliency.reshape(self.grid_size,self.grid_size)
				#saliency /= saliency.sum()
				saliency /= saliency.max()
				seq_dist = list()
				seqs = list()
				out = list()

				for seq in self.raw_seq[key][img_idx]:
					shape = seq.shape
					flag = False # smaller than min dist
					mini_seq = list()
					seq_saliency = np.random.uniform(low=0, high=0.01, size=(self.grid_size, self.grid_size))
					old_fix = (int(self.grid_size * seq[0][0]), int(self.grid_size * seq[0][1]))
					for fix in seq:
						h = int(self.grid_size * fix[0])
						w = int(self.grid_size * fix[1])
						fix = (h,w)
						dist = self._euc(fix, old_fix)
						if dist > self.max_dist:
							flag = True
							break
						elif dist > self.min_dist:
							mini_seq.append(fix)
							seq_saliency[fix[0]][fix[1]] = 255
							old_fix = fix 
                            # if len(mini_seq) == self.seq_len:
                            #   self.sequences[key].append(np.array(mini_seq, dtype=np.int16))
                            #   self._map[key].append((img_idx, len(self.sequences[key]) -1 ))
                            #   mini_seq = list()
					
					if not flag:
						l = len(mini_seq)
						if (l <= self.max_seq) and (l >= self.min_seq):
							seqs.append(np.array(mini_seq, dtype=np.int16))
							#seq_saliency /= seq_saliency.sum()
							#seq_dist.append(entropy(saliency, seq_saliency.reshape(-1)))
							#seq_saliency /= seq_saliency.max()
							seq_saliency = gaussian_filter(seq_saliency, sigma=self.sigma)
							seq_saliency /= seq_saliency.max()
							seq_dist.append(self._euc(saliency, seq_saliency.reshape(-1)))
							#seq_saliency /= seq_saliency.sum()
							#seq_dist.append(entropy(saliency, seq_saliency.reshape(-1))) 
							out.append(seq_saliency)
							#print seq_dist[-1], seq_saliency.mean(), seq_saliency.min(), seq_saliency.max(), saliency.mean(), saliency.min(), saliency.max()
							#self.sequences[key].append(np.array(mini_seq, dtype=np.int16))
							#self._map[key][l].append((img_idx, len(self.sequences[key]) -1))
				seq_dist, seqs = (list(t) for t in zip(*sorted(zip(seq_dist, seqs), key=lambda item:item[0])))
				out = [saliency, seq_dist, out]
				pickle.dump(out, open('pkl/{0}.pkl'.format(img_idx), 'w'))
	

	def _next_lenght(self, batch_size=2, mode='train'):
		tmp = list()
		for key in self._map[mode]:
			l = len(self._map[mode][key]) - self.index[mode][key]
			if l >= batch_size:
				tmp += [key for i in range(l)]

		if len(tmp) == 0:
			for key in self.index[mode]:
				self.index[mode][key] = 0
			return key

		return random.sample(tmp, 1)[0]


	def next_batch(self, batch_size=2, mode='train', norm='L1'):
		batch = list()
		try:
			l = self._next_lenght(batch_size, mode)
			for i in range(batch_size):
				index = self.index[mode][l]
				img_idx , seq_idx = self._map[mode][l][index]
				raw_seq = self.sequences[mode][seq_idx][:,:2]

				seq = list() #processed

				for idx, fix in enumerate(raw_seq):
						z = np.random.uniform(low=0, high=0.1, size=(self.grid_size, self.grid_size))
						z[fix[0]][fix[1]] = 255
						z = gaussian_filter(z, self.sigma)
						if norm=='L1':
							seq.append(z / z.sum())
						else:
							seq.append(z)

				if mode=='train':
					if self.images[mode][img_idx] is not None:
						batch.append([self.images[mode][img_idx], np.array(seq, dtype=np.float16)])
					else:
						img = Image.open(self.stim_path[mode][img_idx])
						if img.mode != 'RGB':
							img = img.convert('RGB')
						img = self.img_processor(img)
						self.images[mode][img_idx] = img
						batch.append([img, np.array(seq, dtype=np.float16)])
				elif mode=='test':
					img = Image.open(self.stim_path[mode][img_idx])
					batch.append([img, np.array(seq, dtype=np.float16)])


				# updating index
				self.index[mode][l] += 1
				# if self.index[mode] >= len(self.stim_path[mode]):
				# 	self.index[mode] = 0

		except Exception as x:
			print(x)

		return batch





	def _euc(self, fix_1, fix_2):

		fix_1 = np.array(fix_1)
		fix_2 = np.array(fix_2)

		return np.power(fix_1 - fix_2, 2).sum()
