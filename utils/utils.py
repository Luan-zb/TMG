import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.  #从给定的索引列表中按顺序采样元素，无需替换
	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):  #img, label---->TOAD
	img = torch.cat([item[0] for item in batch], dim = 0)
	
	label = torch.LongTensor([item[1] for item in batch])
	#print("img label shape",img.shape,label.shape)
	return [img, label]

def collate_features(batch):  #img, coords-->CLAM
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1,num_workers=0):
	#kwargs = {'num_workers': 4, 'pin_memory': False,'num_workers': num_workers} if device.type == "cuda" else {}
	kwargs = {'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 

	ref：https://blog.csdn.net/weixin_36670529/article/details/115249892
		return either the validation loader or training loader
		SequentialSampler:这个可迭代对象是由range产生的顺序数值序列，也就是说迭代是按照顺序进行的
		RandomSampler:根据是否在初始化中给出replacement参数决定是否重复采样，区别核心在于randint()函数生成的随机数学列是可能包含重复数值的，
					  而randperm()函数生成的随机数序列是绝对不包含重复数值的
		SubsetSequentialSampler:Subset Random Sampler应该用于训练集、测试集和验证集的划分，下面将data划分为train和val两个部分，再次指出__iter__()返回的的不是索引，而是索引对应的数据
	"""
	kwargs = {'num_workers': 0} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:  #training=true
				weights = make_weights_for_balanced_classes_split(split_dataset)
				print("helle 1")
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	#https://blog.csdn.net/tyfwin/article/details/108435756
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:  # # validation,test
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def get_optim(model, args):  # ok
	##参数p赋值的元素从列表model.parameters()里取。所以只取param.requires_grad = True（模型参数的可导性是true的元素），就过滤掉为false的元素。
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	elif args.opt == 'adamax':
		optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'agsd':
		optimizer = optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'Adadelta':
		optimizer = optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)	
	elif args.opt == 'Adagrad':
		optimizer = optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)	
	elif args.opt == 'RMSprop':
		optimizer = optim.ASGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):   # ok
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error

def make_weights_for_balanced_classes_split(dataset): # according the sample_size,give the weight to the per class 
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		print("hello 2")
		y = dataset.getlabel(idx)   #self.slide_data['label'][ids]-->represent the 0-7                       
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module): # ok (initialize the weigth+b value)
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)