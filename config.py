

CONFIG = {
	'dataset' : {
			'name': 'OSIE',
			'blur_sigma' : 3,
			'first_blur_sigma': 0,

			'min_sequence_length': 3,
			'max_sequence_length': 10,
			'foveation_radius': 40,
			'sequence_distance': 40 * 2,
			'mask_th' : 0.01,
		}

	'model': {
			'name' : 'DVGG16_CLSTM4',
			'type' : 'RNN',
		},

	'train' : {
		'users': [1],
		'lr' : 3e-4,
		'weight-decay': 1e-4,
		'momentum': 0.9,
		'eval_count': 50,
		'metrics': ['DTW', 'levenshtein_distance', 'frechet_distance', 'hausdorff_distance'],
		},

	'eval' : {
		'users' : range(4),
		'mask_th' : 0.3,
		'next_frame_policy': 'max', # same
		'metrics': ['DTW', 'levenshtein_distance', 'frechet_distance', 'hausdorff_distance', 'MultiMatch'],
		}

	'saliency_train' : range(600),
	'saliency_eval' : range(600,700),


	'weights' : '/media/ramin/data/scanpath/weights/',
	'visualization': '/media/ramin/data/scanpath/visualization/',
}
