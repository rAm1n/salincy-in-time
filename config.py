

CONFIG = {
	'name' : 'OSIE',
	'train' : range(9,15),
	'test' : range(15),
	'saliency_train' : range(600),
	'saliency_test' : range(600,700),
	'blur_sigma' : 3,
	'first_blur_sigma': 0,
	'gaussian_sigma' : 20,
	'mask_th' : 0.01,
	'test_mask_th' : 0.5,
	'distance': 150,
	'encoder' : 'dvgg16',
	'dataset_dir': '/media/ramin/data/scanpath/dataset/OSIE-2/',
}
