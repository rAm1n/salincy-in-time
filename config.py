

CONFIG = {
	'dataset' : {
			'name': 'OSIE',
			'blur_sigma' : 3,
			'first_blur_sigma': 3,

			'min_sequence_length': 3,
			'max_sequence_length': 7,
			'foveation_radius': 30,
			'sequence_distance': 30,
			'mask_th' : 0.01,

			# encoder training
			'saliency_train' : range(600),
			'saliency_test' : range(600,700),
		},

	'model': {
			'name' : 'DVGG16_AttnCLSTM1-64',
			'type' : 'RNN',
		},

	'train' : {
		'users': range(3,8),
		'lr' : 3e-4,
		'weight_decay': 1e-4,
		'momentum': 0.9,
		'eval_count': 50,
		'landa' : 1/7500,
#		'metrics': ['DTW', 'levenshtein_distance', 'frechet_distance',\
#				 'hausdorff_distance','MultiMatch', 'ScanMatch',\
#				 'time_delay_embedding_distance'],
		'metrics': ['DTW', 'levenshtein_distance', 'frechet_distance'],
#		'metrics': [],
		},

	'eval' : {
		'users' : range(3,8),
		'mask_th' : 0.5,
		'next_frame_policy': 'same_norm', # same
		'metrics': ['DTW', 'levenshtein_distance', 'frechet_distance',\
				 'hausdorff_distance','MultiMatch', 'ScanMatch',\
				 'time_delay_embedding_distance'],
		# 'metrics': ['MultiMatch'],
#		'metrics': [],
		},


	'weights_path' : '/media/ramin/data/scanpath/weights/same_norm/',
	'eval_path': '/media/ramin/data/scanpath/eval/same_norm/',
	'visualization_path' : '/media/ramin/data/scanpath/visualization-4/'
}


#MODELS = ['DVGG16_CLSTM1-32', 'DVGG16_CLSTM1-64', 'DVGG16_CLSTM2', 'DVGG16_CLSTM4', 'DVGG16_BCLSTM3']
#MODELS = ['DVGG16_CLSTM2', 'DVGG16_CLSTM4', 'DVGG16_BCLSTM3']


MODELS = ['DVGG16_ATTNCLSTM1-64']

