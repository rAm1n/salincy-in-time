


from __future__ import print_function, division
import os
import torch
import numpy as np
from scipy.spatial import distance
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
from saliency.dataset import SaliencyDataset
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageFilter
from config import CONFIG

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
		# transforms.RandomResizedCrop(224),
#		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])


sal_transform = transforms.Compose([
		# transforms.RandomResizedCrop(224),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])


sal_gt_transform = transforms.Compose([
		transforms.Resize((75,100)),
		transforms.ToTensor(),
	])



class SequnceDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, config, mode='train',  transform=transform, sal_tf=sal_gt_transform):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.config = config
		self.transform = transform
		self.sal_tf = sal_tf
		self.dataset = self.load(mode)


	def __len__(self):
		return len(self.dataset)

	def __repr__(self):
		return 'Dataset object - {0}'.format(self.config['name'])

	def __str__(self):
		return 'Dataset object - {0}'.format(self.config['name'])


	def load(self, mode):
		try:
			dataset = list()

			d = SaliencyDataset(self.config['name'])
			seqs = d.get('sequence')
			imgs = d.get('stimuli_path')
			maps = d.get('heatmap_path')

			for idx , img in enumerate(imgs):
				for seq in seqs[idx][self.config[mode]]:
					if (seq.shape[0] < 3):
						continue
					dataset.append((img, maps[idx], seq))

			return dataset
			# return sorted(dataset, key=lambda k: random.random())

		except OSError as e:
				raise e


	def _prep(self, pair):

		foveated_imgs = list()
		gts = list()

		img , sal, user_seq = pair
		img = Image.open(img)
		sal = Image.open(sal)

		if self.config['first_blur_sigma']:
			foveated_imgs.append(img.filter(ImageFilter.GaussianBlur(self.config['first_blur_sigma'])))
		else:
			foveated_imgs.append(img)

		bl = img.filter(ImageFilter.GaussianBlur(self.config['blur_sigma']))

		user_seq = user_seq[:,[1,0]].astype(np.int32)
		first_fix = [0,0]
		for sec_fix in user_seq:
			try:
				if distance.euclidean(first_fix, sec_fix) < self.config['distance']:
					first_sec = sec_fix
					continue
				blurred = np.array(bl)
				gt = np.zeros(img.size[::-1])
				gt[sec_fix[0], sec_fix[1]] = 2550
				gt = gaussian_filter(gt, self.config['gaussian_sigma'])
				mask = (gt > self.config['mask_th'])
				blurred[mask] = np.array(img)[mask]
				gt[mask] = 255.0

				foveated_imgs.append(Image.fromarray(blurred))
				gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((100,75)), dtype=np.float32) / 255.0
				gts.append(gt)

				first_sec = sec_fix
			except Exception as e:
				print(e)

		return [foveated_imgs, sal, np.array(gts)]

	def __getitem__(self, idx):
		result = list()
		fov, sal, gts = self._prep(self.dataset[idx])
		for img in fov:
			if self.transform:
				img = self.transform(img)
			result.append(img)

		if self.sal_tf:
			sal = self.sal_tf(sal)

		return [torch.stack(result), sal, torch.from_numpy(gts), self.dataset[idx]]




class Saliency(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, config, mode='train', transform=sal_transform, sal_transform=sal_gt_transform):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.config = config
		self.transform = transform
		self.sal_transform = sal_transform
		self.dataset = self.load(mode)


	def __len__(self):
		return len(self.dataset)

	def __repr__(self):
		return 'Dataset object - {0}'.format(self.config['name'])

	def __str__(self):
		return 'Dataset object - {0}'.format(self.config['name'])


	def load(self, mode):
		try:
			dataset = list()

			d = SaliencyDataset(self.config['name'])
			maps = d.get('heatmap_path')[ self.config['saliency_' + mode]]
			imgs = d.get('stimuli_path')[ self.config['saliency_' + mode]]

			for idx , img in enumerate(imgs):
				dataset.append((img,maps[idx]))

			return sorted(dataset, key=lambda k: random.random())

		except OSError as e:
				raise e


	def __getitem__(self, idx):
		img, sal = self.dataset[idx]
		img = Image.open(img)
		sal = Image.open(sal)

		if self.transform:
			img = self.transform(img)
		if self.sal_transform:
			sal = self.sal_transform(sal)

		return [img, sal]
