# TN-MIL-GAP
We have successfully addressed the challenging problem of whole slide image (WSI) classification. WSI produces images of high resolution, which are rich in details; however, it lacks localized annotations.When only slide-level labels are available, WSI classification can be treated as a multiple instance learning (MIL) problem. In this study, we introduce a novel approach for WSI classification, which leverages the MIL and Transformer models. Notably, our proposed method obviates the necessity for localized annotations. Our method consists of three key components. Firstly, we use ResNet50, which has been pre-trained on ImageNet, as an instance feature extractor. Secondly, we present a Transformer-based MIL aggregator that adeptly captures contextual information within individual regions and correlation information among diverse regions within the WSI. Our proposed approach effectively mitigates the issue of high computational complexity in the Transformer architecture by integrating linear attention. Thirdly, we introduce the global average pooling (GAP) layer to increase the mapping relationship between WSI features and category features, further improving classification accuracy. To evaluate our model, we conducted experiments on the CPTAC dataset. The results demonstrate the superiority of our approach compared to previous MIL-based methods. Our proposed method achieves state-of-the-art performance in WSI classification without reliance on localized annotations. Overall, our work offers a robust and effective approach that overcomes challenges posed by high-resolution WSIs and limited annotation availability.

![Image text](https://github.com/Luan-zb/TN-MIL-GAP/blob/main/Figure/Figure1_v1.png)
## Data
The pathology slides and corresponding labels for WSIs are available from the TCIA CPTAC Pathology Portal（https://cancerimagingarchive.net/datascope/cptac/）.
Various types of data are as follows:

Types|Quantity
:---:|:---:
Lung|2162
Skin|388
Kidney|735
Uterus Endometrium|334
Pancreas|469
Soft tissue|294
Head Neck|816
Brain|447

## Conda Environment
- Python (3.7.7)
- h5py (2.10.0)
- matplotlib (3.1.1)
- numpy (1.18.1)
- opencv-python (4.1.1.26)
- openslide-python (1.1.2)
- pandas (0.25.3)
- pillow (7.0.0)
- PyTorch (1.13.0)
- scikit-learn (1.0.2)
- scipy (1.4.1)
- tensorboardx (1.14.0)
- torchvision (0.1.8)
- performer_pytorch (1.1.4)
- einops (0.4.1)
- spams (2.6.1)
- staintools (2.1.2)
- pyyaml (3.13)

## Training
python train.py --drop_out --early_stopping --lr  2e-4 --k 1 --exp_code mil_concat_transformer  --task dummy_mtl_concat  --log_data  --results_dir results --data_root_dir DATA_ROOT_DIR

### optional arguments:
 
 ```
   -h, --help            show this help message and exit
   --drop_out            enabel dropout (p=0.25)
   --early_stopping      enable early stopping　
   --k K                 number of folds (default: 10)
   --exp_code EXP_CODE   experiment code for saving results<br>  
   --task {mil_concat_transformer}
   --log_data            log data using tensorboard
   --results_dir RESULTS_DIR
                        results directory (default: ./results)
   --data_root_dir DATA_ROOT_DIR
                        data directory    
```



## Evaluation
python eval.py --drop_out --k 1 --models_exp_code mil_concat_transformer_s1 --save_exp_code mil_concat_transformer_eval --task origin_predicition  --results_dir results --data_root_dir DATA_ROOT_DIR
### optional arguments:
 ```                    
   -h, --help show this help message and exit
   --drop_out            whether model uses dropout
   --k K                 number of folds (default: 1)
   --models_exp_code MODELS_EXP_CODE
                        experiment code to load trained models (directory
                        under results_dir containing model checkpoints
   --save_exp_code SAVE_EXP_CODE
                        experiment code to save eval results
   --task {origin_predicition}      
   --results_dir RESULTS_DIR
                        relative path to results folder, i.e. the directory
                        containing models_exp_code relative to project root
                        (default: ./results)
   --data_root_dir DATA_ROOT_DIR
                        data directory
```


## Funding
This work was supported by the Strategic Priority Research Program of the Chinese Academy of Sciences (grant number XDB38040100) and the National Natural Science Foundation of China [grant numbers 92259101 and 31771466].


