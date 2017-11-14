
from dataset import *
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import pickle
from scipy.ndimage.filters import gaussian_filter
import random




def make_batch(path='', d_name='SALICON', im_size=(224,224), size=9000, batch_size=4, min_len=5, max_len=30, grid_size=32, gamma=1):
	d = SaliencyBundle(d_name)
	seq = list(d.get('sequence', percentile=True, modify='remove'))[:size]
	stim_path = d.get('stimuli_path')[:size]

	normalize = transforms.Normalize(
	   mean=[0.485, 0.456, 0.406],
	   std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
	   transforms.Scale(im_size),
	   transforms.ToTensor(),
	   normalize
	])


	images = list()
	sequence = list()
	main = {i:list() for i in range(min_len, max_len + 1)}
	dataset = list()
	#########################################
	# removing images with only 1 channel.
	print('starting processs.')
	print('stage 1 - cleaning dataset')
	for img_idx, img in enumerate(stim_path):
		img = Image.open(img)
		if img.mode == 'RGB':
			p = preprocess(img)
			# threshold on seq length
			tmp = list()
			for s in seq[img_idx]:
				shape = s.shape
				if (shape[0] >= min_len) and (shape[0] <= max_len):
					z = np.zeros((shape[0], grid_size, grid_size))
					s = s[:,:2]
					for idx, row in enumerate(s):
						w = int(grid_size * row[0])
						h = int(grid_size * row[1])
						z[idx][h][w] = 1
						z[idx] = gaussian_filter(z[idx], gamma)
					tmp.append(s)
					main[shape[0]].append((p,z))
			images.append(p)
			sequence.append(np.array(tmp))

	print('making batch')
	for key in main.keys():
		l = len(main[key])
		batch = list()
		for i in range(l):
			idx = random.randrange(len(main[key]))
			batch.append(main[key].pop(idx))
			if len(batch) == batch_size:
				dataset.append(batch)
				batch=list()

	random.shuffle(dataset)
	if path:
		print('dumping data')
		f = open(path, 'w')
		pickle.dump(dataset, f)
		f.cloes()
	return dataset
