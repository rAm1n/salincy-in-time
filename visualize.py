from PIL import Image
from utils.salicon import Salicon 
import numpy as np

d = Salicon(gamma=2)
d.initialize()


path = 'test/'

s = d.next_batch()

for idx, I in enumerate(s[0][1]):
	I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
	img = Image.fromarray(I8)
	img = img.resize((224,224))
	img.save(path + "{0}.png".format(idx))

I = s[0][0].permute(1,2,0).cpu().numpy()
I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
img = Image.fromarray(I8)
img.save('test/main.jpg')
