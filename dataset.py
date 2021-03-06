



from __future__ import print_function, division
import os
import shutil
import glob
import pickle
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
from scipy.spatial import distance
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageFilter
import skimage.transform

from saliency.dataset import SaliencyDataset
from config import CONFIG
from utils import fov_mask

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
		# transforms.RandomResizedCrop(224),
#		transforms.RandomHorizontalFlip(),
		# transforms.Pad((0,4,0,4)),
		# transforms.Resize((300,400)),
		transforms.ToTensor(),
		normalize,
	])


sal_transform = transforms.Compose([
		# transforms.RandomResizedCrop(224),
		#transforms.RandomHorizontalFlip(),
		# transforms.Pad((0,4,0,4)),
		transforms.ToTensor(),
		normalize,
	])


sal_gt_transform = transforms.Compose([
		# transforms.Resize((76,100)),
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
		# return sum(x is not None for x in self.dataset)

	def __repr__(self):
		return 'Dataset object - {0}'.format(self.config['dataset']['name'])

	def __str__(self):
		return 'Dataset object - {0}'.format(self.config['dataset']['name'])


	def load(self, mode):
		try:
			dataset = list()

			d = SaliencyDataset(self.config['dataset']['name'])
			seqs = d.get('sequence')[:600]
			imgs = d.get('stimuli_path')[:600]
			maps = d.get('heatmap_path')[:600]


			for img_idx , img in enumerate(imgs):
				for user_idx, seq in enumerate(seqs[img_idx][self.config[mode]['users']]):
					if (seq.shape[0] < self.config['dataset']['min_sequence_length']):
						dataset.append(None)
					else:
						#dataset.append((img, maps[img_idx], seq, [img_idx, self.config[mode]['users'][user_idx] ]))
						dataset.append((img, maps[img_idx], seqs[img_idx], [img_idx, self.config[mode]['users'][user_idx] ]))

			return dataset
			# return sorted(dataset, key=lambda k: random.random())

		except OSError as e:
				raise e


	def _prep(self, pair):

		if pair is None:
			return None

		foveated_imgs = list()
		gts = list()

		img , sal, seqs, [img_idx, user_idx] = pair

		# result = self.check_exists(img_idx, user_idx)
		# if result:
		# 	return result

		img = Image.open(img)#.resize((400,300))
		#seqs = seqs / 2

		w,h = img.size
		sal = Image.open(sal)
		user_seq = seqs[user_idx][:,[0,1]].astype(np.int32)

		# sal.save(os.path.join(self.config['dataset_dir'], '{0}_{1}_{2}.jpg'.format(img_idx, user_idx, 'sal')))
		# sal_copy_path = os.path.join(self.config['dataset_dir'], '{0}_{1}_{2}.jpg'.format(img_idx, user_idx, 'sal'))
		# shutil.copy2(sal, sal_copy_path)


		if self.config['dataset']['first_blur_sigma']:
			foveated_imgs.append(img.filter(ImageFilter.GaussianBlur(self.config['dataset']['first_blur_sigma'])))
		else:
			foveated_imgs.append(img)

		# im_ptrn = os.path.join(self.config['dataset_dir'],
		# 	'{0}_{1}_{2}.jpg'.format(img_idx, user_idx, len(foveated_imgs) -1 ))
		# foveated_imgs[-1].save(im_ptrn)

		bl = np.array(img.filter(ImageFilter.GaussianBlur(self.config['dataset']['blur_sigma'])))
		img = np.array(img)


		first_fix = [0,0]
		fixations = list()
		history = [np.zeros((75,100))]

		for t , sec_fix in enumerate(user_seq):
			try:
				if distance.euclidean(first_fix, sec_fix) < self.config['dataset']['sequence_distance']:
					first_sec = sec_fix
					continue
				if len(foveated_imgs) > self.config['dataset']['max_sequence_length']:
					break
				if (sec_fix[0] > w) or (sec_fix[1] > h):
					continue

				blurred = bl.copy()

				# gt = np.zeros(img.size[::-1])
				# gt[sec_fix[0], sec_fix[1]] = 2550
				# gt = gaussian_filter(gt, self.config['gaussian_sigma'])
				# mask = (gt > self.config['mask_th'])
				mask, gt = fov_mask((h,w), radius=self.config['dataset']['foveation_radius'],
								 	center=sec_fix, th=self.config['dataset']['mask_th'])


				blurred[mask] = img[mask]
				# gt[mask] = 255.0

				foveated_imgs.append(Image.fromarray(blurred))
				# im_ptrn = os.path.join(self.config['dataset_dir'], '{0}_{1}_{2}.jpg'.format(img_idx, user_idx, len(foveated_imgs) -1 ))
				# foveated_imgs[-1].save(im_ptrn)

				# gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((100,75)), dtype=np.float32) / 255.0
				gt = skimage.transform.resize(gt, (75,100))
				gts.append(gt)

				if t > 0:
					cur_history = history[-1] + gts[-1]
					cur_history[cur_history > 1.0] = 1.0
					history.append(cur_history)

				fixations.append(sec_fix)
				first_sec = sec_fix
			except Exception as e:
				pass

		fixations = np.array(fixations)

		# return [foveated_imgs, np.array(gts, dtype=np.float32), sal, fixations, np.array(history)]
		return [foveated_imgs, np.array(gts, dtype=np.float32), sal, seqs, np.array(history)]


	def __getitem__(self, idx):
		try:
			result = list()
			img_idx, user_idx = self.dataset[idx][-1]
			fov, gts, sal, fixations, history = self._prep(self.dataset[idx])
			for img in fov:
				if self.transform:
					img = self.transform(img)
				result.append(img)

			if self.sal_tf:
				sal = self.sal_tf(sal)

			result =  {
				'input' : torch.stack(result),
				'gts'   : torch.from_numpy(gts),
				'saliency' : sal,
				'img_path' : self.dataset[idx][0],
				'fixations': fixations,
				'history' : torch.from_numpy(history).float(),
				'user_idx': user_idx,
				'img_idx' : img_idx,
			}

			return result

		except (ValueError, TypeError) as e:
			print(e)
			return None



class Saliency(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, config, mode='train', transform=transform, sal_transform=sal_gt_transform):
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

			d = SaliencyDataset(self.config['dataset']['name'])
			maps = d.get('heatmap_path')[ self.config['dataset']['saliency_' + mode]]
			imgs = d.get('stimuli_path')[ self.config['dataset']['saliency_' + mode]]

			for idx , img in enumerate(imgs):
				dataset.append((img,maps[idx]))

			return sorted(dataset, key=lambda k: random.random())

		except OSError as e:
				raise e


	def __getitem__(self, idx):
		img, sal = self.dataset[idx]

		# bluring input
		img = Image.open(img)
		w, h = img.size
		if self.config['dataset']['first_blur_sigma']:
			blurred = img.filter(ImageFilter.GaussianBlur(self.config['dataset']['first_blur_sigma']))

			# adding random foveation.
			# mask, gt = fov_mask((h,w), radius=self.config['dataset']['foveation_radius'],
			# 				 	center=(np.random.rand(2) * (w , h)), th=self.config['dataset']['mask_th'])


			# img = np.array(img)
			# blurred = np.array(blurred)
			# blurred[mask] = img[mask]
			# blurred = Image.fromarray(blurred.astype(np.uint8))
			img = blurred


		sal = Image.open(sal)


		if self.transform:
			img = self.transform(img)
		if self.sal_transform:
			sal = self.sal_transform(sal)




		return [img, sal]
