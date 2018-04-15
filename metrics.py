
# reference: https://github.com/rAm1n/saliency-metrics/blob/master/metrics.py

import numpy as np
from scipy.misc import imresize
from scipy.stats import entropy
from scipy.misc import imread
import matlab.engine
import time
import os
from utils import extract_img_sequences, extract_model_fixations



# def MultiMatch(eng, model_output, seqs):
# 	try:
# 		model_fixations = extract_model_fixations(model_output)
# 		seqs = extract_img_sequences(seqs)
# 		result = list()
# 		for seq in seqs:
# 			result.append(_MultiMatch(eng, model_fixations, seq))
# 		return np.array(result).squeeze()
# 	except Exception as e:
# 		print(e)
# 		return False

# def ScanMatch(eng, data1, data2):
# 	pass
