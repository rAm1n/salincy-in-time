
from metrics import MultiMatch, make_engine
import numpy as np

from saliency.dataset import SaliencyDataset
from config import CONFIG
from  scipy.spatial.distance import euclidean




def fov_mask(size=(600,800), radius=30, center=None, th=0.01):
	h, w = size
	sq_size = w
	if h > w:
		sq_size = h

	x = np.arange(0, sq_size, 1, float)
	y = x[:,np.newaxis]
	if center is None:
		x0 = y0 = sq_size // 2
	else:
		x0 = center[0]
		y0 = center[1]
		if (x0 > w) or (y0 > h):
			print('center is out of mask')
			x0 = y0 = sq_size // 2

	circle = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / radius**2)[:h,:w]
	mask = (circle > th)

	return mask, circle



def extract_img_sequences(seqs):
	# all the seqs in seq
	try:
		result = list()
		for user in seqs:
			user_seq = user[:,[1,0,2]].astype(np.int32)
			tmp = list()
			first_fix = [0,0]
			for sec_fix in user_seq:
				try:
					if euclidean(first_fix, sec_fix[:2]) < CONFIG['distance']:
						first_sec = sec_fix
						continue
					else:
						tmp.append(sec_fix)
				except Exception as e:
					print(e)
			tmp = np.array(tmp)
			result.append(tmp)
		return np.array(result)
	except Exception as e:
		print(e)



def extract_model_fixations(seq, add_duration=True):
	output = list()
	for t in seq:
		output.append(np.unravel_index(t.argmax(), t.shape))
	output = np.array(output)
	extra_column = np.random.uniform(low=200, high=800, size=(output.shape[0],1))
	return np.hstack((output,extra_column)).astype(np.int32)
