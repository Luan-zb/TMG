 #ok

# ref link: python create_patches_fp.py --source /home/daichuangchuang/Nature/CPTAC/HNSCC --save_dir RESULTS_DIRECTORY/HNSCC --patch_size 256 --seg --patch --stitch
# generate the directory:(1).mask-->png (2).patches(.h5) (3).stitches(.png) (4).process_list_autogen.csv(contains the segment/patching/stitching parameter)
# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd

def stitching(file_path, wsi_object, downscale = 64): #file_path = os.path.join(patch_save_dir, slide_id+'.h5')
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
	### Start Seg Timer
	start_time = time.time()

	# Segment
	WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)

	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None):
	slides = sorted(os.listdir(source))  #get the file list of the source directory
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]  ##array-like
	#slides (df or array-like): array-like structure containing list of slide ids, if df, these ids assumed to be stored under the 'slide_id' column
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params) ##空值使用默认值填充一下，默认值作为新列插入(主要还是以process_list为主)

	#	default_df_dict = {'slide_id': slide_ids, 'process': np.full((total), 1, dtype=np.uint8)}  #指定每张组织切片是否进行处理-->在初始设置参数时，将所有的process=1
	mask = df['process'] == 1
	process_stack = df[mask]
	total = len(process_stack)  #extracted the unprocessed columns

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)  # write the parameter to the process_list_autogen.csv
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))  # processing output
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0  #set the processed slide's 'process' to 0
		slide_id, _ = os.path.splitext(slide)

		#已经处理的slide则会选择直接进行跳过
		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'  # #'status': np.full((total), 'tbp')
			continue


		# Inialize WSI
		full_path = os.path.join(source, slide)
		WSI_object = WholeSlideImage(full_path)  # return the single WSIs

		if use_default_params:  #use the  use_default_params--->use_default_params=False 
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
		else: ##正常不使用设置的默认参数，run
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}
			for key in vis_params.keys():
				current_vis_params.update({key: df.loc[idx, key]}) #  #使用生成的df(use_default_params)中各个参数信息  run
			for key in filter_params.keys():
				current_filter_params.update({key: df.loc[idx, key]})
			for key in seg_params.keys():
				current_seg_params.update({key: df.loc[idx, key]})
			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:  #-1
			if len(WSI_object.level_dim) == 1:    # #level_dim=((47807, 24430), (11951, 6107), (2987, 1526));len(level_dim)=3
				current_vis_params['vis_level'] = 0	
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)   # get_best_level_for_downsample(downsample) ：对给定的下采样因子返回一个下采样级别   2
				current_vis_params['vis_level'] = best_level         # 2

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level          # 2
		print("len(WSI_object.level_dim) ",len(WSI_object.level_dim))
		print("best_level",current_seg_params['seg_level'])

		## keep_ids 与 exclude_ids 均未None
		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:  # skipped
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else: # run
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0: # skipped
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else: # run
			current_seg_params['exclude_ids'] = []

		'''
		self.level_dim = self.wsi.level_dimensions
        wsi.level_dimensions   ------->((47807, 24430), (11951, 6107), (2987, 1526))
		'''
		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue
		
		#update the df parameter
		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		seg_time_elapsed = -1
		if seg:  # seg_level=2
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask: # default TRUE
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)  ##此时的WSI_object为segmentation的结果
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)
			else:
				print(str(slide_id)+".h5"," No such file or directory")

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'   # #修改状态为'status'为'processed' 正常结束时的状态，若运行过程中出错时，则会被设置为already_existed

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')  #--source /home/daichuangchuang/Nature/CPTAC/HNSCC
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')                                     #--patch_size 256 
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')               #--save_dir RESULTS_DIRECTORY/HNSCC

#模板文件的名称传递给--preset参数即可
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')

#从哪个金字塔等级下采样以提取补丁（默认值为0，最高可用分辨率）
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')

parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
	args = parser.parse_args()
	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 
   
	for key, val in directories.items():  #创建所需要的目录
		print("{} : {}".format(key, val))
		if key not in ['source']: # #创建除原始数据文件夹外的文件夹目录
			os.makedirs(val, exist_ok=True)

   #segmentation parameters
   #seg_level：对 WSI 进行分段的下采样级别（默认值：-1，它使用 WSI 中最接近 64 倍下采样的下采样）
	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
    
   #contour filtering parameters
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    
   #segmentation visualization parameters
	vis_params = {'vis_level': -1, 'line_thickness': 250}
    
   #patch parameters
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:# when preset exits,update the parameters to the pretained paramenters
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=args.step_size, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											patch_level=args.patch_level, patch = args.patch,
											process_list = process_list, auto_skip=args.no_auto_skip)
