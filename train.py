from __future__ import print_function
import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils_mtl_concat_transformer import train
from datasets.dataset_mtl_concat import Generic_WSI_MTL_Dataset, Generic_MIL_MTL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    # test/val auc       test/val acc
    all_cls_test_auc = []
    all_cls_val_auc = []
    all_cls_test_acc = []
    all_cls_val_acc = []


    folds = np.arange(start, end)  # 在本次设置中，start=0 end=1
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path='{}/splits_{}.csv'.format(
                                                                             args.split_dir, i))

        # training：3458  validation：147  testing：297
        print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, cls_test_auc, cls_val_auc, cls_test_acc, cls_val_acc = train(datasets, i, args)
        
        all_cls_test_auc.append(cls_test_auc)
        all_cls_val_auc.append(cls_val_auc)
        all_cls_test_acc.append(cls_test_acc)
        all_cls_val_acc.append(cls_val_acc)

        # write results to pkl(results/dummy_mtl_sex_s1)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)   #{'C3N-03227-21': {'slide_id': array('C3N-03227-21', dtype='<U12'), 'cls_prob': array([[1.6796170e-04, 9.9983203e-01]], dtype=float32), 'cls_label': 1}, 'C3N-00855-26': {'slide_id': array('C3N-00855-26', dtype='<U12'), 'cls_prob': array([[1.0699688e-04, 9.9989295e-01]], dtype=float32)

    final_df = pd.DataFrame({'folds': folds, 'cls_test_auc': all_cls_test_auc,
                             'cls_val_auc': all_cls_val_auc, 'cls_test_acc': all_cls_test_acc,
                             'cls_val_acc': all_cls_val_acc})

    # k=1 equal to len(folds)：->go to summary.csv
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


# Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
# split_dir参数在comand命令中未进行相应的指定，只是在代码中进行了定义
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                         + 'instead of infering from the task and label_frac argument (default: None)')


parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd','adamax','agsd','Adadelta','Adagrad','RMSprop'], default='adamax')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str, choices=['dummy_mtl_concat'])
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)  # seed设置为1
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    # flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法
    torch.backends.cudnn.deterministic = True


# the goal:the same input-->the same output
seed_torch(args.seed)
encoding_size = 1024
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'seed': args.seed,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}



'''
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
'''

if args.task == 'dummy_mtl_concat':
     args.n_classes = 8
     '''
        pt file:  features, label, site, sex 
        h5 file:  features, label, site, sex, coords
     '''
     dataset = Generic_MIL_MTL_Dataset(csv_path='dataset_csv/cohort_all_type.csv',
                                       data_dir=os.path.join(args.data_root_dir,'DATA_ROOT_DIR/pt_files/'),
                                       shuffle=False,
                                       seed=args.seed,
                                       print_info=True,
                                       label_dicts=[{'Lung': 0, 'Skin': 1, 'Kidney': 2, 'Uterus Endometrium':3, 'Pancreas':4, 'Soft Tissue':5, 'Head Neck':6, 'Brain':7}],
                                       label_cols=['label'],
                                       patient_strat=False)
else:
     raise NotImplementedError



# reasult_dir
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
    
# seed=1
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)  # dummy_mtl_sex_s1

# split_dir
if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(100)))  #dummy_mtl_concat_100
else:
    args.split_dir = os.path.join('splits', args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

# save the experiment_ settings
#{'num_splits': 1, 'k_start': -1, 'k_end': -1, 'task': 'dummy_mtl_concat', 'max_epochs': 200, 'results_dir': './results', 'lr': 0.0002, 'experiment': 'dummy_mtl_sex', 'reg': 1e-05, 'seed': 1, 'use_drop_out': True, 'weighted_sample': False, 'opt': 'adamax', 'split_dir': 'splits/dummy_mtl_concat_100'}
with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:  # experiment_dummy_mtl_sex.txt
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
