# TN-MIL-GAP
Cancers of unknown primary (CUPs) are metastatic tumours whose primary site cannot be determined; that is, when malignant cells are discovered in the body, but the primary site of the tumour cannot be determined. Identifying the origin of a tumour is the premise and prerequisite for effective treatment in the current cancer diagnostic and treatment systems, which considerably increases the difficulty of diagnosing and treating patients with CUPs whose primary site cannot be found. The traditional methods used to identify the origin of a tumour often include clinical, radiological, and endoscopic examinations. However, these detection methods are not satisfactorily accurate, sensitive, and specific, and do not provide a diagnostic gold standard. To overcome these challenges, we present a Transformer-based algorithm (TN-MIL-GAP) that can predict the origin of primary tumours. We used whole-slide images of tumours with known primary origins to train a model that could identify the origin sites of the primary tumour. In our dependent test set with known primary origins, the model achieved an AUC of 99.66% and a top-1 accuracy of 94.8%, which is a great improvement compared to other state-of-the-art methods, such as AttentionMIL, TOAD, and MIL.
## Data
The pathology slides and corresponding labels for WSIs are available from the TCIA CPTAC Pathology Portal（https://cancerimagingarchive.net/datascope/cptac/）.

- All images are 224x224 pixels (px) at 0.5 microns per pixel (MPP). 
- All images are color-normalized using Macenko's method (http://ieeexplore.ieee.org/abstract/document/5193250/, DOI 10.1109/ISBI.2009.5193250).
- Tissue classes are: **Adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR), colorectal adenocarcinoma epithelium (TUM).**

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
- scikit-learn (0.22.1)
- scipy (1.4.1)
- tensorflow (1.14.0)
- tensorboardx (1.14.0)
- torchvision (0.1.8)
- performer_pytorch (1.1.4)
- einops (0.4.1)
- spams (2.6.1)
- staintools (2.1.2)
- pyyaml (3.13)

## Training


## Evaluation


## Funding
This work was supported by the Strategic Priority Research Program of the Chinese Academy of Sciences (grant number XDB38040100) and the National Natural Science Foundation of China [grant numbers 92259101 and 31771466].


