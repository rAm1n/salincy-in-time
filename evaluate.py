
from metrics import MultiMatch, make_engine
import numpy as np 

from saliency.dataset import SequenceDataset
from config import CONFIG
from  scipy.spatial.distance import euclidean

features = np.load('/media/ramin/data/scanpath/eval/results.npy')

eng = make_engine()



dataset = SequenceDataset(CONFIG['name'])
seq = dataset.get('sequence')

# for epoch in [0,1]:
# 	for user 


def extract_img_sequences(img_idx):
	# seq is global 
	try:
		img = seq[img_idx]
		result = list()
		for user in img:
			user_seq = user[:,[1,0]].astype(np.int32)
			tmp = list()
			first_fix = [0,0]
			for sec_fix in user_seq:
				try:
					if distance.euclidean(first_fix, sec_fix) < CONFIG['distance']:
						first_sec = sec_fix
						continue
					else:
						tmp.append(sec_fix)
				except Exception as e:
					print(e)
			tmp = np.array(tmp)
			result.append(tmp)
		return result
	except Exception as e:
		print(e)