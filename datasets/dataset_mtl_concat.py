from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
from torch.utils.data import Dataset
import h5py
from utils.utils import generate_split, nth

#当axis=1的时候，concat就是行对齐，然后将不同列名称的两张表合并
def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)  #https://blog.csdn.net/qq_42535601/article/details/86523689
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)

class Generic_WSI_MTL_Dataset(Dataset):
	def __init__(self,
		csv_path = None,
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dicts = [{}, {}, {}],
		patient_strat=False,
		label_cols = ['label', 'site', 'sex'],
		patient_voting = 'max',
		filter_dict = {},
		):
		"""
		Args:
			csv_file (string): Path to the dataset csv file.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dicts (list of dict): List of dictionaries with key, value pairs for converting str labels to int for each label column
			label_cols (list): List of column headings to use as labels and map with label_dicts
			filter_dict (dict): Dictionary of key, value pairs to exclude from the dataset where key represents a column name, 
								and value is a list of values to ignore in that column
			patient_voting (string): Rule for deciding the patient-level label
		"""
		self.custom_test_ids = None
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		self.label_cols = label_cols
		self.split_gen = None

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)

		self.label_dicts = label_dicts
		self.num_classes=[len(set(label_dict.values())) for label_dict in self.label_dicts]

		slide_data = self.df_prep(slide_data, self.label_dicts, self.label_cols)
		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes[0])]		
		for i in range(self.num_classes[0]):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes[0])]
		for i in range(self.num_classes[0]):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def filter_df(df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	@staticmethod
	def df_prep(data, label_dicts, label_cols):
		if label_cols[0] != 'label':
			data['label'] = data[label_cols[0]].copy()

		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dicts[0][key]

		for idx, (label_dict, label_col) in enumerate(zip(label_dicts[1:], label_cols[1:])):
			print(label_dict, label_col)
			data[label_col] = data[label_col].map(label_dict)

		return data

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):

		for task in range(len(self.label_dicts)):
			print('task: ', task)
			print("label column: {}".format(self.label_cols[task]))
			print("label dictionary: {}".format(self.label_dicts[task]))
			print("number of classes: {}".format(self.num_classes[task]))
			print("slide-level counts: ", '\n', self.slide_data[self.label_cols[task]].value_counts(sort = False))
		
		for i in range(self.num_classes[0]):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def sample_held_out(self, test_num = (40, 40)):

		test_ids = []
		np.random.seed(self.seed) #fix seed
		
		if self.patient_strat:
			cls_ids = self.patient_cls_ids
		else:
			cls_ids = self.slide_cls_ids

		for c in range(len(test_num)):
			test_ids.extend(np.random.choice(cls_ids[c], test_num[c], replace = False)) # validation ids

		if self.patient_strat:
			slide_ids = [] 
			for idx in test_ids:
				case_id = self.patient_data['case_id'][idx]
				slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
				slide_ids.extend(slide_indices)

			return slide_ids
		else:
			return test_ids

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits=None, split_key='train', return_ids_only=False, split=None):  #train_split = self.get_split_from_df(all_splits, 'train')
		#all_splits = pd.read_csv(splits_0.csv)  # splits_0.csv
		if split is None:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True)  # 会将原来的索引index作为新的一列,使用drop参数设置去掉原索引，获取splits_0.csv文件对应的train/val/test的一列

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())  #将所有的slide在相应split集合中取出来
			if return_ids_only:
				ids = np.where(mask)[0]
				return ids
			'''
		    slide_data = pd.read_csv(csv_path)  五列数据
		    slide_data = self.filter_df(slide_data, filter_dict)
			'''
			df_slice = self.slide_data[mask].dropna().reset_index(drop=True)  #将在slide_data中slide_id在相应的split中记录保留下来
			'''
			label_dicts = [{'Lung':0, 'Skin':1},{'Primary':0,  'Metastatic':1},{'F':0, 'M':1}]
        	label_cols = ['label', 'site', 'sex']
        	num_classes=[len(set(label_dict.values())) for label_dict in label_dicts]-->[2,2,2]
			'''
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)
		else:
			split = None
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):  #from_id=False,csv_path="splits/dummy_mtl_concat_100/splits_0.csv"
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)
			
			else:
				test_split = None
			
		#datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
		else:
			'''
			assert是一种常见的调试方式，对boolean表达式进行检查
				一个正确程序必须保证这个boolean表达式的值为true.
				如果该值为false，说明程序已经处于不正确的状态下，系统将给出警告并且退出
			'''
			assert csv_path   #csv_path为None时，则会报错
			all_splits = pd.read_csv(csv_path)  #splits_0.csv
			train_split = self.get_split_from_df(all_splits, 'train')  #return the object
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids, task):
		if task > 0:
			return self.slide_data[self.label_cols[task]][ids]
		else:
			return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):
		if return_descriptor:
			dfs = []
			for task in range(len(self.label_dicts)):
				index = [list(self.label_dicts[task].keys())[list(self.label_dicts[task].values()).index(i)] for i in range(self.num_classes[task])]
				columns = ['train', 'val', 'test']
				df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)
				dfs.append(df)

		for task in range(len(self.label_dicts)):
			index = [list(self.label_dicts[task].keys())[list(self.label_dicts[task].values()).index(i)] for i in range(self.num_classes[task])]
			for split_name, ids in zip(['train', 'val', 'test'], [self.train_ids, self.val_ids, self.test_ids]):
				count = len(ids)
				print('\nnumber of {} samples: {}'.format(split_name, count))
				labels = self.getlabel(ids, task)
				unique, counts = np.unique(labels, return_counts=True)
				missing_classes = np.setdiff1d(np.arange(self.num_classes[task]), unique)
				unique = np.append(unique, missing_classes)
				counts = np.append(counts, np.full(len(missing_classes), 0))
				inds = unique.argsort()
				counts = counts[inds]
				for u in range(len(unique)):
					print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
					if return_descriptor:
						dfs[task].loc[index[u], split_name] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			df = pd.concat(dfs, axis=0) 
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)

class Generic_MIL_MTL_Dataset(Generic_WSI_MTL_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
		super(Generic_MIL_MTL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle
	'''
	在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[idx]取值。当实例对象做P[idx]运算时，就会调用类中的__getitem__()方法
	'''
	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		#site = self.slide_data[self.label_cols[1]][idx]   #site
		#sex = self.slide_data[self.label_cols[2]][idx]    #sex
		if type(self.data_dir) == dict:   #type(self.data_dir) == str
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			full_path = os.path.join(data_dir, '{}.pt'.format(slide_id))   # Feature_Extraction 包含提取的特征矩阵
			features = torch.load(full_path)  # number(patches number)*1024

			return features, label
			

		else:
			full_path = os.path.join(data_dir, '{}.h5'.format(slide_id))   ## Feature_Extraction 不仅包含提取的特征矩阵，还包含 patch坐标
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]  # number(patches number)* 1024
				coords = hdf5_file['coords'][:]		  #  number(patches number)* 2（代表的是每个patches的左上角坐标）

			features = torch.from_numpy(features)
			return features, label, coords

'''
Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, label_cols=self.label_cols)
'''

class Generic_Split(Generic_MIL_MTL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2, label_cols=None):
		self.use_h5 = False
		self.slide_data = slide_data                                                   #slide_data = pd.read_csv('dataset_csv/CM_LSCC_Processed_Augmentation.csv')
		self.data_dir = data_dir                                                       #Nature/CLAM/DATA_ROOT_DIR/DATA_DIR/pt_files/
		self.num_classes = num_classes                                                 #[2,2,2]-->[{'Lung':0, 'Skin':1},'Primary':0,  'Metastatic':1},{'F':0, 'M':1}]
		self.slide_cls_ids = [[] for i in range(self.num_classes[0])]  #[[], []]
		self.label_cols = label_cols                                   # [label site sex]
		self.infer = False
		for i in range(self.num_classes[0]):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]          #将属于Lung和Skin的slides_id存储到数组中

	def __len__(self):
		return len(self.slide_data)
