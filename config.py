

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
		},

	'model': {
			'name' : 'DVGG16_CLSTM1',
			'type' : 'RNN',
		},

	'train' : {
		'users': [5],
		'lr' : 3e-4,
		'weight_decay': 1e-4,
		'momentum': 0.9,
		'eval_count': 50,
		'landa' : 0.5,
		'metrics': ['DTW', 'frechet_distance', 'hausdorff_distance'],
#		'metrics': ['frechet_distance'],
		'metrics': [],
		},

	'eval' : {
		'users' : [5],
		'mask_th' : 0.5,
		'next_frame_policy': 'same_norm', # same
		#'metrics': ['DTW', 'levenshtein_distance', 'frechet_distance', 'hausdorff_distance', 'MultiMatch'],
		# 'metrics': ['MultiMatch'],
		'metrics': [],
		},

	'saliency_train' : range(600),
	'saliency_eval' : range(600,700),


	'weights_path' : '/media/ramin/data/scanpath/weights/same_norm/',
	'eval_path': '/media/ramin/data/scanpath/eval/same_norm/',
	'visualization_path' : '/media/ramin/data/scanpath/visualization/'
}


#MODELS = ['DVGG16_CLSTM1-32', 'DVGG16_CLSTM1-64', 'DVGG16_CLSTM2', 'DVGG16_CLSTM4', 'DVGG16_BCLSTM3']
#MODELS = ['DVGG16_CLSTM2', 'DVGG16_CLSTM4', 'DVGG16_BCLSTM3']


MODELS = ['DVGG16_CLSTM1-32', 'DVGG16_CLSTM2', 'DVGG16_BCLSTM3']
