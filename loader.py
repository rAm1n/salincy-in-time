


from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import random
import os
import pickle
from utils.config import CONFIG
from urlparse import urljoin




normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])

transform_body = transforms.Composje([
	transforms.ToPILImage(),
	transforms.Resize((256,256)),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	normalize,
	])


class Dataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, config, mode='train',  transform=transform):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.config = config
		self.transform_body = transform_body
		self.transform_full = transform_full
		self.directory = os.path.join(self.config['data_path'], self.name)

		self.dataset = self._load_dataset(mode)


	def __len__(self):
		return len(self.dataset)

	def __repr__(self):
		return 'Dataset object - {0}'.format(self.name)

	def __str__(self):
		return 'Dataset object - {0}'.format(self.name)

	def __len__(self):
		return len(self.data)

	def load(self):
		try:
			if not os.path.isdir(self.directory):
				self.download()
			self._load()
		except OSError as e:
				raise e


	def download(self):
		try:
			os.makedirs(directory)
		except OSError as e:
				raise e

		# downloading base pickle file
		pkl_url = urljoin(self.config['json_directory'],
					 '{0}.pkl'.format(self.name.upper()))
		pkl_path = self._download(pkl_url, extract=False)

		try:
			with open(pkl_path, 'r') as pkl:
				data = pickle.load(pkl)#, encoding='latin1')
				for url in data['url']:
					self._download(url, extract=True)
		except Exception as x:
			print(x)

		#loading data
		self._load()


	def _download(self, url, extract=False):
		try:
			filename = url.split('/')[-1]
			directory = os.path.join(self.config['data_path'], self.name)
			file_path = os.path.join(directory, filename)

			print('downloading - {0}'.format(url))
			wget.download(url, file_path)
			if extract:
				if url[-3:] == 'zip':
					zip_ref = zipfile.ZipFile(file_path, 'r')
					zip_ref.extractall(directory)
					zip_ref.close()
				else:
					tar = tarfile.open(file_path, 'r')
					tar.extractall(directory)
					tar.close()
				os.remove(file_path)
				return directory

			return file_path

		except Exception as x:
			print(x)
			os.rmdir(directory)


	def _load(self):
		pkl_directory = os.path.join(self.config['data_path'], self.name)
		try:
			path = os.path.join(pkl_directory, '{0}.pkl'.format(self.name))
			f = open(path, 'rb')
			data = pickle.load(f)#, encoding='latin1')
			for key,value in data.items():
				setattr(SaliencyBundle, key, value)
			# pre-processing data
			self.len = len(data)
		except Exception as x:
			print(x)



	def _load_dataset(self, mode):

		DIR = self.config['pair_dir']

		pair_1 = self.config['pair_pattern'].format(1, mode, self.config['num_class'])
		pair_2 = self.config['pair_pattern'].format(2, mode, self.config['num_class'])

		pair_1 = os.path.join(DIR, pair_1)
		pair_2 = os.path.join(DIR, pair_2)


		pairs = list()

		with open(pair_1) as pair_1, open(pair_2) as pair_2:
			for x, y in zip(pair_1, pair_2):
				img_1, cls_1 = x.strip().split()
				img_2, cls_2 = y.strip().split()
				if cls_1 != cls_2:
					print('holy fuck!')
					print(img_1, cls_1)
					print(img_2, cls_2)
					exit()
				img_1 = os.path.join(self.config['body_dir'], img_1.split('/')[-1])
				img_2 = os.path.join(self.config['body_dir'], img_2.split('/')[-1])
				img = os.path.join(self.config['img_dir'], img_1.split('/')[-1][3:])

				pairs.append((img, img_1, img_2, int(cls_1)))

		random.shuffle(pairs)
		return pairs


	def __getitem__(self, idx):
		img_full = io.imread(self.dataset[idx][0])
		img_1 = io.imread(self.dataset[idx][1])
		img_2 = io.imread(self.dataset[idx][2])
		cls = self.dataset[idx][3]

		if self.transform_full:
			img_full = self.transform_full(img_full)
		if self.transform_body:
			img_1 = self.transform_body(img_1)
			img_2 = self.transform_body(img_2)

		return [img_full, img_1, img_2, cls]






# TODO add len in json and fix code


class SaliencyBundle():
	def __init__(self, name, config=CONFIG):
		self.name =  name
		self.config = config
		self._download_or_load()

	def __repr__(self):
		return 'Dataset object - {0}'.format(self.name)

	def __str__(self):
		return 'Dataset object - {0}'.format(self.name)

	def __len__(self):
		return len(self.data)

	def _download_or_load(self):

		try:
			directory = os.path.join(self.config['data_path'], self.name)
			self.directory = directory
			if os.path.isdir(directory):
				return self._load()
			os.makedirs(directory)
		except OSError as e:
				raise e

		# Starting Download
		# for url in URLs[self.name]:
		# 	self._download(url, unzip=True)
		pkl_url  = self.config['json_directory'] + self.name.upper() + '.pkl'
		self._download(pkl_url)

		pkl_directory = os.path.join(self.config['data_path'], self.name)
		try:
			path = os.path.join(pkl_directory, '{0}.pkl'.format(self.name))
			f = open(path, 'rb')
			data = pickle.load(f)#, encoding='latin1')
			for url in data['url']:
				self._download(url, extract=True)
			f.close()
		except Exception as x:
			print(x)

		#loading data
		self._load()


	def _download(self, url, extract=False):
		try:
			filename = url.split('/')[-1]
			directory = os.path.join(self.config['data_path'], self.name)
			dst = os.path.join(directory, filename)

			print('downloading - {0}'.format(url))
			wget.download(url, dst)
			if extract:
				if url[-3:] == 'zip':
					zip_ref = zipfile.ZipFile(dst, 'r')
					zip_ref.extractall(directory)
					zip_ref.close()
				else:
					tar = tarfile.open(dst, 'r')
					tar.extractall(directory)
					tar.close()
				os.remove(dst)

		except Exception as x:
			print(x)
			os.rmdir(directory)


	def _load(self):
		pkl_directory = os.path.join(self.config['data_path'], self.name)
		try:
			path = os.path.join(pkl_directory, '{0}.pkl'.format(self.name))
			f = open(path, 'rb')
			data = pickle.load(f)#, encoding='latin1')
			for key,value in data.items():
				setattr(SaliencyBundle, key, value)
			# pre-processing data
			self.len = len(data)
		except Exception as x:
			print(x)

	def get(self, data_type, **kargs):
		result = list()
		for img in tqdm(self.data):
			if data_type=='sequence':
				tmp = list()
				for user in img['sequence']:
					user = np.array(user)
					if 'percentile' in kargs:
						if kargs['percentile']:
							if(user.shape)[0] == 0:
								continue
							_sample = user[:,:2] / self.size
							user = np.concatenate((_sample, user[:,2:]), axis=1)
					if 'modify' in kargs:
						if kargs['modify']== 'fix' :
							if 'percentile' in kargs:
								if kargs['percentile']:
									mask_greater = _sample > 1.0
									mask_smaller = _sample < 0.0
									_sample[mask_greater] = 0.999999
									_sample[mask_smaller] = 0.000001
									user = np.concatenate((_sample, user[:,2:]), axis=1)
								else:
									# TODO
									print('fix was ignored, only works in percentile mode.')
							else:
									# TODO
								print('fix was ignored, only works in percentile mode.')
						elif kargs['modify'] == 'remove':
							if 'percentile' in kargs:
								if kargs['percentile']:
									user = user[user[:,0]<=0.99999, :]
									user = user[user[:,0]>=0.00001, :]
									user = user[user[:,1]<=0.99999, :]
									user = user[user[:,1]>=0.00001, :]
								else:
									# TODO
									print('fix was ignored, only works in percentile mode.')
							else:
								# TODO
								print('fix was ignored, only works in percentile mode.')
					tmp.append(user)
				tmp = np.array(tmp)
					# else:
					# 	tmp = np.array([np.array(user) for user in img['sequence']])

			elif data_type =='heatmap':
				path = os.path.join(self.directory, img['heatmap'])
				print path
				if os.path.isfile(path):
					tmp = imread(path)
				else:
					tmp = np.fromstring( img['heatmap'].decode('base64'), \
						dtype='int8').reshape(self.size)

			elif data_type == 'heatmap_path':
							tmp = os.path.join(self.directory, img['heatmap'])

			elif data_type =='stimuli':
				path = os.path.join(self.directory, img['stimuli'])
				if os.path.isfile(path):
					tmp = imread(path)
			elif data_type == 'stimuli_path':
				tmp = os.path.join(self.directory, img['stimuli'])
			else:
				try:
					tmp = self.data[data_type]
				except Exception as x:
					return False
			result.append(tmp)

		result = np.asarray(result)
		return result



