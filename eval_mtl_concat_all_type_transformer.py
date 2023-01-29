#CUDA_VISIBLE_DEVICES=0 python eval_mtl_concat_all_type.py --drop_out --k 1 --models_exp_code dummy_mtl_sex_s1 --save_exp_code dummy_mtl_sex_s1_eval --task study_v2_mtl_sex  --results_dir results --data_root_dir /home/daichuangchuang/Nature/CLAM/DATA_ROOT_DIR
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
import matplotlib.pyplot as plt
from datasets.dataset_mtl_concat import Generic_MIL_MTL_Dataset, save_splits
import h5py
from utils.eval_utils_mtl_concat import *
import time
# Training settings
parser = argparse.ArgumentParser(description='TOAD Evaluation Script')
parser.add_argument('--data_root_dir', type=str, help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. ' +
                         'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--drop_out', action='store_true', default=False,
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=1, help='number of folds (default: 1)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                    default='test')  
parser.add_argument('--task', type=str, choices=['study_v2_mtl_sex'])

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoding_size = 1024
# save_exp_code：dummy_mtl_sex_s1_eval
# models_exp_code：dummy_mtl_sex_s1
args.save_dir = os.path.join('./eval_results_Transformer_test37', 'EVAL_' + str(args.save_exp_code))  # save the result
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))  # load the model

os.makedirs(args.save_dir, exist_ok=True)
# 若split_dir 为None
if args.splits_dir is None:
    args.splits_dir = args.models_dir  # /results/dummy_mtl_sex_s1

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,  #
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'drop_out': args.drop_out,
            'micro_avg': args.micro_average}

# {'task': 'study_v2_mtl_sex', 'split': 'test', 'save_dir': './eval_results/EVAL_dummy_mtl_sex_s1_eval', 'models_dir': 'results/dummy_mtl_sex_s1', 'drop_out': True, 'micro_avg': False}
with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:  # ./eval_results/EVAL_dummy_mtl_sex_s1_eval/eval_experiment_dummy_mtl_sex_s1_eval.txt
    print(settings, file=f)
f.close()
print(settings)


if args.task == 'study_v2_mtl_sex':
    args.n_classes = 8
    '''
       pt file:  features, label, site, sex
       h5 file:  features, label, site, sex, coords
    '''
    dataset = Generic_MIL_MTL_Dataset(csv_path='dataset_csv/cohort_all_type.csv',
                                      data_dir=os.path.join(args.data_root_dir,'/data/luanhaijing/project/tissue_process_pipeline/DATA_ROOT_DIR/pt_files'),
                                      shuffle=False,
                                      print_info=True,
                                      label_dicts=[{'Lung': 0, 'Skin': 1, 'Kidney': 2, 'Uterus Endometrium':3, 'Pancreas':4, 'Soft Tissue':5, 'Head Neck':6, 'Brain':7}],
                                      label_cols=['label'],
                                      patient_strat=False)
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)  # start=0, end=1, folds=0
else:
    folds = range(args.fold, args.fold + 1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]  # results/dummy_mtl_sex_s1/s_0_checkpoint.pt

datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_cls_auc = []
    all_cls_acc = []
    all_cls_top3_acc = []
    all_cls_top5_acc = []

    for ckpt_idx in range(len(ckpt_paths)):  # 0
        print("ckpt_idx",ckpt_idx)
        if datasets_id[args.split] < 0:  # split default:test
            split_dataset = dataset  # features, label, site, sex
            csv_path = None
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir,
                                                 folds[ckpt_idx])  ##results/dummy_mtl_sex_s1/splits_0.csv
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)  # train_split, val_split, test_split

            split_dataset = datasets[datasets_id[args.split]]  # 返回对象  当使用split_dataset[idx]方法时，则可以获取相应的返回值
            '''
                pt file:  features, label, site, sex
                h5 file:  features, label, site, sex, coords
             '''
        
        #total_time=[0 for x in range(100)]
        #for i in range(0,100):
        #    time_start=time.time()
        #    model, results_dict = eval(split_dataset, args, ckpt_paths[ckpt_idx])  # s_0_checkpoint.pt
        #    time_elapsed=time.time()-time_start
        #    total_time[i]=time_elapsed
        

        model, results_dict = eval(split_dataset, args, ckpt_paths[ckpt_idx])  # s_0_checkpoint.pt

        #print("evaluating the slides tooks(average) {} s".format(np.mean(total_time)))
        #print("evaluating the slides tooks(var) {} s".format(np.var(total_time)))
        #print("evaluating the slides tooks(std) {} s".format(np.std(total_time)))
        #print("results_dict",results_dict)
        '''
        results_dict {'patient_results': {'C3L-00086-21': {'slide_id': array('C3L-00086-21', dtype='<U12'), 'cls_prob': array([[7.5928263e-05, 8.9621803e-09, 1.9781267e-07, 9.9992001e-01,
        3.7969528e-06, 2.5020865e-13, 8.2527576e-09, 1.9852151e-13]],
        dtype=float32), 'cls_label': 3}....},'cls_test_error': 0.0707070707070707, 'cls_auc': 0.9968677793225678, 'cls_aucs': [0.9918990411109886, 0.9984070091596974, 0.9977110419337119, 0.9991633199464525, 0.9960049937578028, 0.9922268907563025, 0.9996389891696751, 0.9998909487459107], 'loggers': <utils.core_utils_mtl_concat.Accuracy_Logger object at 0x7fb492e79490>, 'df':          slide_id    Y  Y_hat           p_0           p_1           p_2           p_3           p_4           p_5           p_6           p_7
        0    C3L-00086-21  3.0      3  7.592826e-05  8.962180e-09  1.978127e-07  9.999200e-01  3.796953e-06  2.502087e-13  8.252758e-09  1.985215e-13
        ..            ...  ...    ...           ...           ...           ...           ...           ...           ...           ...           ...
        296  C3L-01681-26  2.0      2  3.557418e-08  1.424291e-07  9.999961e-01  2.537523e-11  1.062955e-06  1.628036e-07  6.343173e-12  2.594927e-06
        [297 rows x 11 columns], 'top1_acc': 0.9292929172515869, 'top3_acc': 0.996632993221283, 'top5_acc': 0.996632993221283}
        '''

        for cls_idx in range(len(results_dict['cls_aucs'])):  # len(results_dict['cls_aucs']) 代表类别数目
            print('class {} auc: {}'.format(cls_idx, results_dict['cls_aucs'][cls_idx]))

        all_cls_auc.append(results_dict['cls_auc'])
        all_cls_acc.append(1 - results_dict['cls_test_error'])
        # all_site_auc.append(results_dict['site_auc'])
        # all_site_acc.append(1-results_dict['site_test_error'])
        # all_cls_top3_acc.append(results_dict['top3_acc'])
        # all_cls_top5_acc.append(results_dict['top5_acc'])

        df = results_dict['df']
        
        '''
        297个测试样本
        slide_id    Y  Y_hat           p_0           p_1           p_2           p_3           p_4           p_5           p_6           p_7
        C3L-00086-21  3.0      3  7.592826e-05  8.962180e-09  1.978127e-07  9.999200e-01  3.796953e-06  2.502087e-13  8.252758e-09  1.985215e-13
        ........................................................................................................................................
        C3L-01681-26  2.0      2  3.557418e-08  1.424291e-07  9.999961e-01  2.537523e-11  1.062955e-06  1.628036e-07  6.343173e-12  2.594927e-06
        '''
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])),index=False)  # eval_results/EVAL_dummy_mtl_sex_s1_eval/fold_0.csv

    df_dict = {'folds': folds, 'cls_test_auc': all_cls_auc, 'cls_test_acc': all_cls_acc}
    final_df = pd.DataFrame(df_dict)  # eval_results/EVAL_dummy_mtl_sex_s1_eval/summary.csv
    '''
    ,folds,cls_test_auc,cls_test_acc
    0,0,1.0,1.0
    '''
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
