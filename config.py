

CONFIG = {
	'name' : 'OSIE',
	'train' : [1],
	'test' : range(4),
	'saliency_train' : range(600),
	'saliency_test' : range(600,700),
	'blur_sigma' : 3,
	'first_blur_sigma': 0,
	'min_sequence_length': 3,
	'foveation_radius': 40,
	'sequence_distance': 40 * 2,
	'max_sequence_length': 10,
	'gaussian_sigma' : 20,
	'mask_th' : 0.01,
	'test_mask_th' : 0.3,
	'encoder' : 'dvgg16',
	'dataset_dir': '/media/ramin/data/scanpath/dataset/OSIE-2/',
}
