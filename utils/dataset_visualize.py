
import scipy.ndimage
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as g
import glob
from PIL import Image
import time




slot_time = 500
n_imgs = 50
grid_factor = 8

times = range(slot_time, 5000, slot_time)
slots = len(times)
#result = np.zeros((n_imgs, slots, int(480/grid_factor) + 1, int(640/grid_factor) + 1 ), dtype=np.float32)
#result = np.zeros((n_imgs, slots, 480+1, 640+1), dtype=np.float32)
result = np.zeros((n_imgs, 481, 641), dtype=np.float32)

imgs = glob.glob('train/*.mat')[:n_imgs]



for img_idx, img in enumerate(imgs):
	mat = scipy.io.loadmat(img)
	for user in mat['gaze']:
		location = user[0]['location']
		timestamp = user[0]['timestamp']
		for s_idx, slot in enumerate(times):
#			end = slot
#			start = slot - slot_time
#			prev = 0
			for r_idx,record in enumerate(timestamp):
				item = location[r_idx]
				result[img_idx, int(item[1]), int(item[0])] = 255
#				if (record[0] > start) and (record[0] < end):
#					item = location[r_idx]
#					result[img_idx, s_idx, int(item[1]), int(item[0])] = 255
	#				result[img_idx, s_idx, int(item[1]/ grid_factor), int(item[0]/ grid_factor)] += 1
	#				prev = record[0]


#for img_idx, img in enumerate(imgs):
#	for t in range(result.shape[1]):
#		m = result[img_idx, t].max()
#		result[img_idx, t] = (result[img_idx, t] / m) *255
#		mask = result[img_idx, t] < 100
#		result[img_idx, t][mask] = 0
	



fig = plt.figure()
base = 331
t = 1
i = 30
#for t in range(9):
#	plt.subplot(base + t )
#	print(result[i,t].sum(), result[i,t].max())
#	plt.imshow(g(result[i,t],1), interpolation=None, cmap='gray')
			

#plt.show()
#out = np.zeros((481, 641))
#for t in result:
#	out+=t 
#plt.figure(2)

#for i in range(18, n_imgs):
#	img = '/Users/Ramin/Desktop/images/' + imgs[i].split('/')[-1].split('.mat')[0] + '.jpg'
#	img = Image.open(img).resize((641,481))
#	img.show()
#	time.sleep(3)

img = '/Users/Ramin/Desktop/images/' + imgs[i].split('/')[-1].split('.mat')[0] + '.jpg'
img = Image.open(img).resize((641,481))

mask = Image.fromarray(result[3].astype(np.uint8)).convert('RGB')

new_img = Image.blend(img, mask, alpha=0.6).convert('RGB')

plt.imshow(new_img, interpolation=None, cmap='gray')

#plt.imshow(result[3], interpolation=None, cmap='gray')


