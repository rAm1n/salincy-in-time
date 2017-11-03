


import json
import wget
from scipy.misc import imread
import numpy as np
import zipfile
import tarfile
import os
from tqdm import tqdm




CONFIG = {
	'data_path' : os.path.expanduser('~/tmp/saliency/'),
	'auto_download' : True,
	'json_directory' : 'http://saliency.raminfahimi.ir/json/'
}


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
		json_url  = self.config['json_directory'] + self.name.upper() + '.json'
		self._download(json_url)

		json_directory = os.path.join(self.config['data_path'], self.name)
		try:
			path = os.path.join(json_directory, '{0}.json'.format(self.name))
			f = open(path, 'r')
			data = json.load(f)
			for url in data['url']:
				self._download(url, extract=True)
			f.close()
		except Exception,x:
			print x

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

		except Exception,x:
			print x
			os.rmdir(directory)


	def _load(self):
		json_directory = os.path.join(self.config['data_path'], self.name)
		try:
			path = os.path.join(json_directory, '{0}.json'.format(self.name))
			f = open(path, 'r')
			data = json.load(f)
			for key,value in data.iteritems():
				setattr(SaliencyBundle, key, value)
			# pre-processing data
			self.len = len(data)
		except Exception,x:
			print x

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
				if os.path.isfile(path):
					tmp = imread(path)
				else:
					tmp = np.fromstring( img['heatmap'].decode('base64'), \
						dtype='int8').reshape(self.size)
			elif data_type =='stimuli':
				path = os.path.join(self.directory, img['stimuli'])
				if os.path.isfile(path):
					tmp = imread(path)
			elif data_type == 'stimuli_path':
				tmp = os.path.join(self.directory, img['stimuli'])
			else:
				try:
					tmp = self.data[data_type]
				except Exception,x:
					return False
			result.append(tmp)

		result = np.asarray(result)
		return result



