

from dataset import *
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms


d_name = 'CAT2000'
d = SaliencyBundle(d_name)
seq = list(d.get('sequence', percentile=True, modify='remove'))
stim_path = d.get('stimuli_path')



d_name = 'CAT2000'
im_size = (224,224)
batch_size = 4
min_len = 5
max_len = 31


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
#########################################
# removing images with only 1 channel.
print('starting processs.')
print('stage 1 - cleaning dataset')
for idx, img in enumerate(stim_path):
	print(idx)
	img = Image.open(img)
	if img.mode == 'RGB':
		p = preprocess(img)
		# threshold on seq length
		tmp = list()
		for s in seq[idx]:
			shape = s.shape
			if (shape[0] > min_len) or (shape[0] < max_len):
				tmp.append(s)
		images.append(p)
		sequence.append(np.array(tmp))


images = np.array(images)
