from __future__ import print_function
import argparse
import os
import shutil
import time
import numpy as np
import sys
from datetime import datetime
import logging
import cv2
import time
import scipy.misc
from PIL import Image
import gc
import skvideo.io
import glob

from metrics import MultiMatch, make_engine
import numpy as np

from saliency.dataset import SaliencyDataset
from config import CONFIG
from  scipy.spatial.distance import euclidean

# features = np.load('/media/ramin/data/scanpath/eval/results.npy')
#
# eng = make_engine()
#
#
#
# dataset = SaliencyDataset(CONFIG['name'])
# seq = dataset.get('sequence')

# for epoch in [0,1]:
# 	for user


def extract_img_sequences(seq, img_idx):
	# all the seqs in seq
	try:
		img = seq[img_idx]
		result = list()
		for user in img:
			user_seq = user[:,[1,0]].astype(np.int32)
			tmp = list()
			first_fix = [0,0]
			for sec_fix in user_seq:
				try:
					if euclidean(first_fix, sec_fix) < CONFIG['distance']:
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



def extract_model_fixations(seq):
	output = list()
	for t in seq:
		output.append(np.unravel_index(t.argmax(), t.shape))
	return output
