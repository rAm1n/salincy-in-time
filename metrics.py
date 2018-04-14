
# reference: https://github.com/rAm1n/saliency-metrics/blob/master/metrics.py

import numpy as np
from scipy.misc import imresize
from scipy.stats import entropy
from scipy.misc import imread
import matlab.engine
import time
import os
from utils import extract_img_sequences, extract_model_fixations



"""
Collection of common saliency metrics

If you're using this code, please don't forget to cite the original code
as mentioned in each function doc.

"""


def NSS(saliency_map, fixation_map):
	""""
	normalized scanpath saliency between two different
	saliency maps as the mean value of the normalized saliency map at
	fixation locations.

		Computer NSS score.
		:param saliency_map : predicted saliency map
		:param fixation_map : ground truth saliency map.
		:return score: float : score

	"""
	if isinstance(saliency_map, np.ndarray):
		saliency_map = np.array(saliency_map)
	if isinstance(fixation_map, np.ndarray):
		fixation_map = np.array(fixation_map)

	if saliency_map.size != fixation_map.size:
		saliency_map = imresize(saliency_map, fixation_map.shape)


	MAP = (saliency_map - saliency_map.mean()) / (saliency_map.std())
	mask = MAP.astype(bool)

	score =  MAP[mask].mean()

	return score


def CC(sal_map_1, sal_map_2):
	"""
	This finds the linear correlation coefficient between two different
	saliency maps (also called Pearson's linear coefficient).
	score=1 or -1 means the maps are correlated
	score=0 means the maps are completely uncorrelated

	saliencyMap1 and saliencyMap2 are 2 real-valued matrices

		Computer CC score .
		:param sal_map_1 : first saliency map
		:param sal_map_2 : second  saliency map.
		:return score: float : score

	"""
	if isinstance(sal_map_1, np.ndarray):
		sal_map_1 = np.array(sal_map_1, dtype=np.float32)
	elif sal_map_1.dtype != np.float32:
		sal_map_1 = sal_map_1.astype(np.float32)

	if isinstance(sal_map_2, np.ndarray):
		sal_map_2 = np.array(sal_map_2, dtype=np.float32)
	elif sal_map_1.dtype != np.float32:
		sal_map_2 = sal_map_2.astype(np.float32)

	if sal_map_1.size != sal_map_2.size:
		sal_map_1 = imresize(sal_map_1, sal_map_2.shape)

	sal_map_1 = (sal_map_1 - sal_map_1.mean()) / (sal_map_1.std())
	sal_map_2 = (sal_map_2 - sal_map_2.mean()) / (sal_map_2.std())

	score = np.corrcoef(sal_map_1,sal_map_2)

	return score


def EMD():
	"""

	if you are using this function, please cite the following papers:

		http://www.ariel.ac.il/sites/ofirpele/fastemd/code/
		https://github.com/wmayner/pyemd

	"""
	pass




def KLdiv(saliency_map , fixation_map):
	"""
	This finds the KL-divergence between two different saliency maps when
	viewed as distributions: it is a non-symmetric measure of the information
	lost when saliencyMap is used to estimate fixationMap.

		Computer KL-divergence.
		:param saliency_map : predicted saliency map
		:param fixation_map : ground truth saliency map.
		:return score: float : score

	"""

	if isinstance(saliency_map, np.ndarray):
		saliency_map = np.array(saliency_map, dtype=np.float32)
	elif saliency_map.dtype != np.float32:
		saliency_map = saliency_map.astype(np.float32)

	if isinstance(fixation_map, np.ndarray):
		fixation_map = np.array(fixation_map, dtype=np.float32)
	elif fixation_map.dtype != np.float32:
		fixation_map = fixation_map.astype(np.float32)

	# the function will normalize maps before computing Kld
	score = entropy(saliency_map, fixation_map)
	return score



def _AUC(saliency_map, fixation_map, step_size=.01, Nrand=100000):
	"""
		please cite:  https://github.com/NUS-VIP/salicon-evaluation

		Calculates AUC score.
		:param salinecy_map : predicted saliency map
		:param fixation_map : ground truth saliency map.
		:return score: int : score

	"""
	saliency_map = fixation_map - np.min(fixation_map)
	if np.max(saliency_map) > 0:
		saliency_map = saliency_map / np.max(saliency_map)

	S = saliency_map.reshape(-1)
	Sth = np.asarray([ saliency_map[y-1][x-1] for y,x in saliency_map ])

	Nfixations = len(saliency_map)
	Npixels = len(S)

	# sal map values at random locations
	randfix = S[np.random.randint(Npixels, size=Nrand)]

	allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),step_size)
	allthreshes = allthreshes[::-1]
	tp = np.zeros(len(allthreshes)+2)
	fp = np.zeros(len(allthreshes)+2)
	tp[-1]=1.0
	fp[-1]=1.0
	tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
	fp[1:-1]=[float(np.sum(randfix >= thresh))/Nrand for thresh in allthreshes]

	score = np.trapz(tp,fp)
	return score


def SAUC(saliency_map, fixation_map, shuf_map=np.zeros((480,640)), step_size=.01):
	"""
		please cite:  https://github.com/NUS-VIP/salicon-evaluation

		calculates shuffled-AUC score.

		:param salinecy_map : predicted saliency map
		:param fixation_map : ground truth saliency map.
		:return score: int : score

	"""

	saliency_map = fixation_map - np.min(fixation_map)
	if np.max(saliency_map) > 0:
		saliency_map = saliency_map / np.max(saliency_map)
	Sth = np.asarray([ saliency_map[y-1][x-1] for y,x in fixation_map ])
	Nfixations = len(fixation_map)

	others = np.copy(shuf_map)
	for y,x in fixation_map:
		others[y-1][x-1] = 0

	ind = np.nonzero(others) # find fixation locations on other images
	nFix = shuf_map[ind]
	randfix = saliency_map[ind]
	Nothers = sum(nFix)

	allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),step_size)
	allthreshes = allthreshes[::-1]
	tp = np.zeros(len(allthreshes)+2)
	fp = np.zeros(len(allthreshes)+2)
	tp[-1]=1.0
	fp[-1]=1.0
	tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
	fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

	score = np.trapz(tp,fp)
	return scrore



def AUC(salMap, fixMap):
	"""Computes AUC for given saliency map 'salMap' and given
	fixation map 'fixMap'"""
	fixMap = (fixMap>0.7).astype(int)
	salShape = salMap.shape
	fixShape = fixMap.shape
	predicted = salMap.reshape(salShape[0]*salShape[1], -1,
							   order='F').flatten()
	actual = fixMap.reshape(fixShape[0]*fixShape[1], -1,
							order='F').flatten()
	labelset = np.arange(2)
	auc = area_under_curve(predicted, actual, labelset)
	return auc


def area_under_curve(predicted, actual, labelset):
	tp, fp = roc_curve(predicted, actual, np.max(labelset))
	auc = auc_from_roc(tp, fp)
	return auc

def auc_from_roc(tp, fp):
	h = np.diff(fp)
	auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
	return auc

def roc_curve(predicted, actual, cls):
	si = np.argsort(-predicted)
	tp = np.cumsum(np.single(actual[si]==cls))
	fp = np.cumsum(np.single(actual[si]!=cls))
	tp = tp/np.sum(actual==cls)
	fp = fp/np.sum(actual!=cls)
	tp = np.hstack((0.0, tp, 1.0))
	fp = np.hstack((0.0, fp, 1.0))
	return tp, fp


def make_engine():
	return matlab.engine.start_matlab()


def _MultiMatch(eng, data1, data2, check=False):
	try:
		data1 = matlab.double(data1.tolist())
		data2 = matlab.double(data2.tolist())
		if (check) and ('metrics/MultiMatchToolbox' not in eng.pwd()) :
			eng.cd('metrics/MultiMatchToolbox/')
		return eng.doComparison(data1,data2)
	except Exception as e:
		print(e)
		return False


def MultiMatch(eng, model_output, seqs):
	try:
		model_fixations = extract_model_fixations(model_output)
		seqs = extract_img_sequences(seqs)
		result = list()
		for seq in seqs:
			result.append(_MultiMatch(eng, model_fixations, seq))
		return np.array(result).squeeze()
	except Exception as e:
		print(e)
		return False

def ScanMatch(eng, data1, data2):
	pass
