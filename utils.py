
import numpy as np

from saliency.dataset import SaliencyDataset
from config import CONFIG
from  scipy.spatial.distance import euclidean
import skimage.transform




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
			print(' x0 : ', x0, ' w : ', w,' y0: ', y0, 'h : ', h)
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



def extract_model_fixations(seq, size):
	output = list()
	for t in seq:
		t = skimage.transform.resize(t, (size[0],size[1]))
		output.append(np.unravel_index(t.argmax(), t.shape)[::-1])
	return np.array(output, dtype=np.int32)
