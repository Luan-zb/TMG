from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from PIL import Image
# from utils.eval_utils_mtl_concat import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.TransformerMIL_LeFF_NystromAttention import TransMIL
from models.TransformerMIL_LeFF_MSA import TransformerMIL
from models.model_toad import TOAD_fc_mtl_concat
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
import openslide

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config.yaml")
args = parser.parse_args()

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	print("feature.shape:",features.shape)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB,TransMIL,TOAD_fc_mtl_concat,TransformerMIL)):
			#model_results_dict = model(features)
			print("------------------------------------")
			#print("model_results_dict:",model_results_dict)
			'''
			model_results_dict: {'logits': tensor([[ -8.2309,  -9.0179,  -5.5014,  -6.0045, -12.7447, -12.1216,  -6.1124,
			-19.4449]], device='cuda:0'), 'Y_prob': tensor([[2.9072e-02, 1.3234e-02, 4.4555e-01, 2.6939e-01, 3.1854e-04, 5.9398e-04,
			2.4184e-01, 3.9202e-07]], device='cuda:0'), 'Y_hat': tensor([[2]], device='cuda:0'), 'A': tensor([[4.4440, 8.9769, 7.0494,  ..., 4.2777, 8.1451, 5.3196]],
			device='cuda:0')}
	        '''
			# logits, Y_prob, Y_hat, A, _ = model(features)
			results_dict= model(features)
			Y_hat = results_dict['Y_hat'].item()
			print("Y_hat:",Y_hat)
			Y_prob=results_dict['Y_prob']
			A=results_dict['A']
			

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	#Y_hat: Kidney, Y: Kidney, Y_prob: ['0.0291', '0.0132', '0.4455', '0.2694', '0.0003', '0.0006', '0.2418', '0.0000']


		probs, ids = torch.topk(Y_prob, k)
		print("probs, ids:",probs, ids)  #probs, ids: tensor([[4.4555e-01, 2.6939e-01, 2.4184e-01, 2.9072e-02, 1.3234e-02, 5.9398e-04,3.1854e-04, 3.9202e-07]], device='cuda:0') tensor([[2, 3, 6, 0, 1, 5, 4, 7]], device='cuda:0')

		probs = probs[-1].cpu().numpy()
		print("probs:",probs)            #probs: [4.4554600e-01 2.6939422e-01 2.4184154e-01 2.9071651e-02 1.3233662e-02 5.9397798e-04 3.1854049e-04 3.9202166e-07]

		ids = ids[-1].cpu().numpy()
		print("ids:",ids)				#ids: [2 3 6 0 1 5 4 7]
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])
		print("preds_str:",preds_str)   #preds_str: ['Kidney' 'Uterus Endometrium' 'Head Neck' 'Lung' 'Skin' 'Soft Tissue'  'Pancreas' 'Brain']



	return ids, preds_str, probs, A

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict

if __name__ == '__main__':
	config_path = os.path.join('heatmaps/configs', args.config_file)
	config_dict = yaml.safe_load(open(config_path, 'r'))
	print("config_dict",config_dict)
	config_dict = parse_config_dict(args, config_dict)
	print("config_dict:",config_dict)

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))
			
	# decision = input('Continue? Y/N ')
	# if decision in ['Y', 'y', 'Yes', 'yes']:
	# 	pass
	# elif decision in ['N', 'n', 'No', 'NO']:
	# 	exit()
	# else:
	# 	raise NotImplementedError

	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
	model_args.update({'n_classes': args['exp_arguments']['n_classes']})
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
	sample_args = argparse.Namespace(**args['sample_arguments'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])                   #256 x 256
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))  #128 x 128
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))  #patch_size: 256 x 256, with 0.50 overlap, step size is 128 x 128

	
	preset = data_args.preset
	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	# if preset is not None:
	# 	preset_df = pd.read_csv(preset)
	# 	for key in def_seg_params.keys():
	# 		def_seg_params[key] = preset_df.loc[0, key]

	# 	for key in def_filter_params.keys():
	# 		def_filter_params[key] = preset_df.loc[0, key]

	# 	for key in def_vis_params.keys():
	# 		def_vis_params[key] = preset_df.loc[0, key]

	# 	for key in def_patch_params.keys():
	# 		def_patch_params[key] = preset_df.loc[0, key]


	if data_args.process_list is None:
		if isinstance(data_args.data_dir, list):
			slides = []
			for data_dir in data_args.data_dir:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(data_args.data_dir))
		slides = [slide for slide in slides if data_args.slide_ext in slide]
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
		
	else:
		df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
		print("df:",df)
		'''
		slide_id   label
		0  TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A01...  Kidney
		'''
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	ckpt_path = model_args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	
	if model_args.initiate_fn == 'initiate_model':
		model =  initiate_model(model_args, ckpt_path)
	else:
		raise NotImplementedError


	feature_extractor = resnet50_baseline(pretrained=True)
	feature_extractor.eval()

	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Done!')

	label_dict =  data_args.label_dict
	print('label_dict:',label_dict)
	class_labels = list(label_dict.keys())
	print('class_labels:',class_labels)
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
	print('reverse_label_dict:',reverse_label_dict)   #{0: 'Lung', 1: 'Skin', 2: 'Kidney', 3: 'Uterus Endometrium', 4: 'Pancreas', 5: 'Soft Tissue', 6: 'Head Neck', 7: 'Brain'}


	if torch.cuda.device_count() > 1:
		device_ids = list(range(torch.cuda.device_count()))
		print("device_ids:",device_ids)  # [0,1]
		feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
		print("feature_extractor:",feature_extractor)
	else:
		feature_extractor = feature_extractor.to(device)

	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	for i in range(len(process_stack)):
		slide_name = process_stack.loc[i, 'slide_id']
		print("slide_name",slide_name)
		if data_args.slide_ext not in slide_name:
			slide_name+=data_args.slide_ext
		print('\nprocessing: ', slide_name)	
		try:
			label = process_stack.loc[i, 'label']
		except KeyError:
			label = 'Unspecified'
		slide_id = slide_name.replace(data_args.slide_ext,'')  #TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9
		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label  #Kidney

		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)   #/data/luanhaijing/project/tissue_process_pipeline/heatmaps/heatmap_production_results/HEATMAP_OUTPUT/Kidney


		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)   #/data/luanhaijing/project/tissue_process_pipeline/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/Kidney/TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9/


		if heatmap_args.use_roi:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
		else:  #run
			top_left = None
			bot_right = None
		
		print('slide id: ', slide_id)
		print('top left: ', top_left, ' bot right: ', bot_right)
		if isinstance(data_args.data_dir, str):    # run
			slide_path = os.path.join(data_args.data_dir, slide_name)
		elif isinstance(data_args.data_dir, dict):
			data_dir_key = process_stack.loc[i, data_args.data_dir_key]
			slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
		else:
			raise NotImplementedError
		
		mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
		
		# Load segmentation and filter parameters
		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()
		
		# through the process_stack parameter for the experiment params
		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		vis_params = load_params(process_stack.loc[i], vis_params)
		
		keep_ids = str(seg_params['keep_ids'])
		if len(keep_ids) > 0 and keep_ids != 'none':
			seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		else:
			seg_params['keep_ids'] = []
		
		exclude_ids = str(seg_params['exclude_ids'])
		if len(exclude_ids) > 0 and exclude_ids != 'none':
			seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		else:
			seg_params['exclude_ids'] = []
		
		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))
		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))
		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))

		print('Initializing WSI object')
		wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
		print('Done!')
		
		wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
		

		wsi=openslide.open_slide(slide_path)
		seg_level=2
		img1 = np.array(wsi.read_region((0, 0),seg_level, wsi.level_dimensions[seg_level]).convert("RGB"))
		img1 = Image.fromarray(img1)
		img1.save("original_correct.jpg", quality=100)
		print("slide_path:",slide_path)
		print("original_correct wsi.level_dimensions[seg_level]):",wsi.level_dimensions[seg_level])


		#after generate the wsi (error:right-bottom)
		vis_level=2
		img2 = np.array(wsi_object.wsi.read_region((0,0), vis_level, wsi_object.wsi.level_dimensions[vis_level]).convert("RGB"))
		img2 = Image.fromarray(img2)
		img2.save("original0.jpg", quality=100)
		print("original0.jpg wsi_object.wsi.level_dimensions[vis_level]:",wsi_object.wsi.level_dimensions[vis_level])

		print("patch_level",patch_args.patch_level)       # 0
		print("wsi_ref_downsample",wsi_ref_downsample)    # (1.0,1.0)
		print("patch_args.custom_downsample",patch_args.custom_downsample)  #1
		print("patch_size:",patch_size)                   # patch_size: (256, 256)

		# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
		vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))
		
		block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
		mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
		if vis_params['vis_level'] < 0:
			best_level = wsi_object.wsi.get_best_level_for_downsample(32)
			vis_params['vis_level'] = best_level
		mask = wsi_object.visWSI(**vis_params, number_contours=True)
		mask.save(mask_path)
		
		features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
		h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
		
		##### check if h5_features_file exists ######
		#############################################
		if not os.path.isfile(h5_path) :          #TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9.h5 save the coords+features
			_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
	 										model=model,   #CLAM、TOAD
	 										feature_extractor=feature_extractor, 
	 										batch_size=exp_args.batch_size, **blocky_wsi_kwargs, #batch_size：384
	 										attn_save_path=None, feat_save_path=h5_path, 
	 										ref_scores=None)				
		
		##### check if pt_features_file exists ######
		if not os.path.isfile(features_path):     #TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9.pt save the features
			file = h5py.File(h5_path, "r")
			features = torch.tensor(file['features'][:])
			torch.save(features, features_path)
			file.close()
		
		# load features (set the length of the feature size)
		features = torch.load(features_path)
		process_stack.loc[i, 'bag_size'] = len(features)
		
		#############################################
		wsi_object.saveSegmentation(mask_file)  #TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_mask.pkl
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
		print("===========================")
		print("A shape:",A.shape)   #A shape: (47818, 1)

		del features
		
		if not os.path.isfile(block_map_save_path):   #TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_blockmap.h5 save the coords+attention_scores
			file = h5py.File(h5_path, "r")  #TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9.h5 save the coords+features
			coords = file['coords'][:]
			file.close()
			asset_dict = {'attention_scores': A, 'coords': coords}
			block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
		# # save top 3 predictions
		# for c in range(exp_args.n_classes):
		# 	process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
		# 	process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]
			
		os.makedirs('heatmaps/results/', exist_ok=True)
		if data_args.process_list is not None:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)  #heatmap_demo_dataset.csv
		else:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
		
	# 获取排名前10的patch图片及坐标信息，以及得分等
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		print("=scores shape=:",scores.shape)  #=scores shape=: (47818, 1)
		coords = coord_dset[:]
		file.close()

		
	# get the first 15  topk high attention(patches)
	# 	samples = sample_args.samples
	# 	for sample in samples:
	# 		if sample['sample']:
	# 			tag = "label_{}_pred_{}".format(label, Y_hats[0])
	# 			sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
	# 			os.makedirs(sample_save_dir, exist_ok=True)    #heatmaps/heatmap_production_results/HEATMAP_OUTPUT/sampled_patches/label_Kidney_pred_0/topk_high_attention
	# 			print('sampling {}'.format(sample['name']))
	# 			sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
	# 				score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
	# 			for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
	# 				print('coord: {} score: {:.3f}'.format(s_coord, s_score))
	# 				patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
	# 				patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))
		
		wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
	 	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}
		print("wsi_kwargs:",wsi_kwargs)  #wsi_kwargs: {'top_left': None, 'bot_right': None, 'patch_size': (256, 256), 'step_size': (128, 128), 'custom_downsample': 1, 'level': 0, 'use_center_shift': True}

		 
		
		# get the heatmap
		heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
		if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
			pass
		else:
    		#############################################
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
	 						thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
			heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.tif'.format(slide_id)))  #TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_blockmap.png
			del heatmap
		
		save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))  #TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_0.5_roi_False.h5
		if heatmap_args.use_ref_scores:   #true
			ref_scores = scores
		else:
			ref_scores = None
		
		if heatmap_args.calc_heatmap:  #true
			##############################################
			compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs, 
								attn_save_path=save_path,  ref_scores=ref_scores)
		
		print("heatmap_args.use_roi:",heatmap_args.use_roi)    #heatmap_args.use_roi: False
		if not os.path.isfile(save_path):
			print('heatmap {} not found'.format(save_path))
			if heatmap_args.use_roi:   #false
				save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		file = h5py.File(save_path, 'r')     ##TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_0.5_roi_False.h5
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		print("-coord_dset -:",coord_dset.shape)   #-coord_dset -: (190953, 2)
		print("-scores shape-:",scores.shape)      #-scores shape-: (190953, 1)  
		coords = coord_dset[:]
		file.close()




		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
		print("heatmap_vis_args:",heatmap_vis_args)
		if heatmap_args.use_ref_scores:  #use_ref_scores: true
			heatmap_vis_args['convert_to_percentiles'] = False
		print("heatmap_vis_args['convert_to_percentiles']:",heatmap_vis_args['convert_to_percentiles'])  #heatmap_args.use_roi: False

			
		
		heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), 
																						#  overlap: 0.5
																						int(heatmap_args.use_roi),
																						#  # whether to only compute heatmap for ROI specified by x1, x2, y1, y2 ---->use_roi: false 
	 																					int(heatmap_args.blur), 
																						#  # whether to use gaussian blur for further smoothing ---->blur: false
	 																					int(heatmap_args.use_ref_scores),
																						 #  # whether to calculate percentile scores in reference to the set of non-overlapping patches ---->use_ref_scores: true
																						int(heatmap_args.blank_canvas),  
																						#whether to use a blank canvas instead of original slide ---->blank_canvas: false
	 																					float(heatmap_args.alpha), 
																						 #transparency for overlaying heatmap on background (0: background only, 1: foreground only) ---->alpha: 0.4
																						int(heatmap_args.vis_level), 
																						#downsample at which to visualize heatmap (-1 refers to downsample closest to 32x downsample) ---->vis_level: 1
	 																					int(heatmap_args.binarize),
																						#whether to binarize attention scores ---->binarize: false
																						float(heatmap_args.binary_thresh),
																						#  # binarization threshold: (0, 1) ---->binary_thresh: -1
																						heatmap_args.save_ext)
		#TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_0.5_roi_0_blur_0_rs_1_bc_0_a_0.4_l_1_bi_0_-1.0.jpg
		print("slide_path:",slide_path)
		if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
			pass
		else:
			print("--scores shape--:",scores.shape)
			##############################################
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,
				cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
				binarize=heatmap_args.binarize, 
				blank_canvas=heatmap_args.blank_canvas,
				thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
				overlap=patch_args.overlap, 
				top_left=top_left, bot_right = bot_right)
			
			'''
			drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
	 						thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
			'''
			if heatmap_args.save_ext == 'jpg':   #jpg
			#TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_0.5_roi_0_blur_0_rs_1_bc_0_a_0.4_l_1_bi_0_-1.0.jpg
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
			else:
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))	
				
		# generate the origin H&E image
		if heatmap_args.save_orig:  #  # whether to also save the original H&E image --->save_orig: true
			if heatmap_args.vis_level >= 0:   #downsample at which to visualize heatmap (-1 refers to downsample closest to 32x downsample)--->vis_level: 1
				vis_level = heatmap_args.vis_level
			else:
				vis_level = vis_params['vis_level']
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)   
			print("vis_level:",vis_level)												#vis_level: 1
			print("heatmap_args.custom_downsample:",heatmap_args.custom_downsample)     #heatmap_args.custom_downsample: 1            
			#TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_orig_1.jpg
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
				#after generate the wsi (error:left-top)
				if heatmap_args.save_ext == 'jpg':#run
				##TCGA-UW-A7GI-01Z-00-DX1.4A1709FC-F8F3-4509-A011-10CBF1B53DA9_orig_1.jpg
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
	
	print("config_dict:",config_dict)			
	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)