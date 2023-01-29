#ok

import pandas as pd
import numpy as np
import pdb
'''
initiate a pandas df describing a list of slides to process
args:
	slides (df or array-like): 
		array-like structure containing list of slide ids, if df, these ids assumed to be
		stored under the 'slide_id' column
	seg_params (dict): segmentation paramters 
	filter_params (dict): filter parameters
	vis_params (dict): visualization paramters
	patch_params (dict): patching paramters
	use_heatmap_args (bool): whether to include heatmap arguments such as ROI coordinates
'''
def initialize_df(slides, seg_params, filter_params, vis_params, patch_params, 
	use_heatmap_args=False, save_patches=False):

	total = len(slides)
	if isinstance(slides, pd.DataFrame):  ##针对两种不同的数据类型分别进行相应的处理
		slide_ids = slides.slide_id.values
	else:
		slide_ids = slides
	default_df_dict = {'slide_id': slide_ids, 'process': np.full((total), 1, dtype=np.uint8)} ##指定每张组织切片是否进行处理，在初始设置将每个slide的process状态均设置为1

	# initiate empty labels in case not provided
	if use_heatmap_args:
		default_df_dict.update({'label': np.full((total), -1)})  ## 两列数据，每列的size为total   在初始设置将每个slide的label均设置为-1
	
	default_df_dict.update({ #C3L-00081-21.svs,0,processed,2,8,7,4,False,none,none,100.0,16.0,8,2,250,True,four_pt
		'status': np.full((total), 'tbp'),
		# seg params
		'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
		'sthresh': np.full((total), int(seg_params['sthresh']), dtype=np.uint8),
		'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.uint8),
		'close': np.full((total), int(seg_params['close']), dtype=np.uint32),
		'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),
		'keep_ids': np.full((total), seg_params['keep_ids']),
		'exclude_ids': np.full((total), seg_params['exclude_ids']),
		
		# filter params
		'a_t': np.full((total), int(filter_params['a_t']), dtype=np.float32),
		'a_h': np.full((total), int(filter_params['a_h']), dtype=np.float32),
		'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

		# vis params
		'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
		'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

		# patching params
		'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
		'contour_fn': np.full((total), patch_params['contour_fn'])
		})

	if save_patches: # save_patches=False
		default_df_dict.update({
			'white_thresh': np.full((total), int(patch_params['white_thresh']), dtype=np.uint8),
			'black_thresh': np.full((total), int(patch_params['black_thresh']), dtype=np.uint8)})

	if use_heatmap_args:  # use_heatmap_args=False,
		# initiate empty x,y coordinates in case not provided
		default_df_dict.update({'x1': np.empty((total)).fill(np.NaN), 
			'x2': np.empty((total)).fill(np.NaN), 
			'y1': np.empty((total)).fill(np.NaN), 
			'y2': np.empty((total)).fill(np.NaN)})

	if isinstance(slides, pd.DataFrame):##此时的slides代表的是pd.read_csv(process_list)生成的表格，若存在，则使用默认值填充空字段，否则将默认值作为新列插入
		temp_copy = pd.DataFrame(default_df_dict) # temporary dataframe w/ default params  (# 若存在，则使用默认值填充空字段，否则将默认值作为新列插入)
		# find key in provided df
		# if exist, fill empty fields w/ default values, else, insert the default values as a new column
		for key in default_df_dict.keys(): 
			if key in slides.columns:
				mask = slides[key].isna()
				slides.loc[mask, key] = temp_copy.loc[mask, key]
			else:
				slides.insert(len(slides.columns), key, default_df_dict[key])
	else: #run-->RETURN THE SLIDE 
		slides = pd.DataFrame(default_df_dict)
	
	return slides  #return the df-size table