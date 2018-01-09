from PIL import Image
from utils.salicon import Salicon 
import numpy as np

d = Salicon(size=200, sigma=1)
#d.initialize()
d.load()

path = 'test/'

s = d.next_batch(mode='train', norm=None)


I = s[0][0].permute(1,2,0).cpu().numpy()
main = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
main = Image.fromarray(main)
main = main.resize((224,224))

for idx, I in enumerate(s[0][1]):
	I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
	img = Image.fromarray(I8).convert('RGB')
	img = img.resize((224,224))
	Image.blend(main, img, alpha=0.7).save(path + "{0}.png".format(idx))


# img.save('test/main.jpg')
