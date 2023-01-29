#python create_splits.py --task dummy_mtl_concat --seed 1 --k 1
import pdb
import os
import pandas as pd
from datasets.dataset_mtl_concat import Generic_WSI_MTL_Dataset, Generic_MIL_MTL_Dataset, save_splits
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= -1,
										help='fraction of labels (default: [1.0])')
parser.add_argument('--seed', type=int, default=1,
										help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
										help='number of splits (default: 10)')
parser.add_argument('--hold_out_test', action='store_true', default=False,
										help='fraction to hold out (default: 0)')
parser.add_argument('--split_code', type=str, default=None)
parser.add_argument('--task', type=str, choices=['dummy_mtl_concat'],help="which task to run")
args = parser.parse_args()

#how to generate the cohort_all_type_processed.csv?
# import csv
# process_dir="/home/daichuangchuang/project/tissue_region_pipeline/FEATURES_DIRECTORY"
# with open("dataset_csv/cohort_all_type.csv",'w') as f:
#     csvfile=csv.writer(f)
#     csvfile.writerow(["case_id","slide_id","label"])
#     for root,dir,filename in os.walk(process_dir):
#         for name in filename:
#             if name.endswith(".pt"):
#                 abs_path=os.path.join(root,name)
#                 slide_id=abs_path.split("/")[-1][:-3]
#                 case_id=slide_id[:-3]
#                 label=abs_path.split("/")[-3]
#                 csvfile.writerow([case_id,slide_id,label])

if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_WSI_MTL_Dataset(csv_path = 'dataset_csv/cohort_all_type.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts=[{'Lung': 0, 'Skin': 1, 'Kidney': 2, 'Uterus Endometrium':3, 'Pancreas':4, 'Soft Tissue':5, 'Head Neck':6, 'Brain':7}],#label_dicts = [{'Lung': 0, 'Soft Tissue':1}],
                            label_cols = ['label'],
                            patient_strat= False)       
else:
	raise NotImplementedError
#label_dicts = [{'Lung': 0, 'Skin': 1, 'Kidney': 2, 'Uterus Endometrium':3, 'Pancreas':4, 'Soft Tissue':5, 'Head Neck':6, 'Brain':7}]


num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
print("num_slides_cls",num_slides_cls)
val_num = np.floor(num_slides_cls * 0.1).astype(int)
test_num = np.floor(num_slides_cls * 0.2).astype(int)

print("val_num",val_num)    # val_num [43  9 21 24 15  8 10 17]
print("test_num",test_num)  # test_num [86 18 43 48 30 17 20 35]


if __name__ == '__main__':
		if args.label_frac > 0:  # run
			label_fracs = [args.label_frac]
		else:
			label_fracs = [1.0]
        
		if args.hold_out_test:
			custom_test_ids = dataset.sample_held_out(test_num=test_num)
		else: # run
			custom_test_ids = None
		for lf in label_fracs:
			if args.split_code is not None:
				split_dir = 'splits/'+ str(args.split_code) + '_{}'.format(int(lf * 100))
			else:  # run
				split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))  # splits/dummy_mtl_concat_100
			
			#k=1;label_fracs = [1.0];custom_test_ids = None
			dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf, custom_test_ids=custom_test_ids)

			os.makedirs(split_dir, exist_ok=True)
			for i in range(args.k):  #k=1
				if dataset.split_gen is None:
					ids = []
					for split in ['train', 'val', 'test']:
						ids.append(dataset.get_split_from_df(pd.read_csv(os.path.join(split_dir, 'splits_{}.csv'.format(i))), split_key=split, return_ids_only=True))
					dataset.train_ids = ids[0]
					dataset.val_ids = ids[1]
					dataset.test_ids = ids[2]
				else:
					dataset.set_splits()

				descriptor_df = dataset.test_split_gen(return_descriptor=True)
				descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))
				'''
				,train,val,test
				Lung,1217,43,86
				Skin,257,9,18
				Kidney,439,21,43
				Uterus Endometrium,474,24,48
				Pancreas,281,15,30
				Soft Tissue,188,8,17
				Head Neck,207,10,20
				Brain,395,17,35
				'''
				splits = dataset.return_splits(from_id=True)
				save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
				'''
				,train,val,test
				0,C3N-02729-22,C3L-03976-21,C3L-03963-22
				1,C3N-02729-21,C3N-00556-23,C3L-01606-21
				2,C3N-01019-23,C3L-02665-24,C3N-02973-23
				.......................................
				'''
				save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
				'''
				,train,val,test
				C3N-02729-22,True,False,False
				C3N-02729-21,True,False,False
				C3N-01019-23,True,False,False
				C3L-04365-23,True,False,False
				'''