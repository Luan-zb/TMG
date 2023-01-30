# TN-MIL-GAP
Identify various tissues in a pathological section, including but not limited to stroma, lymphocytes, tumor cells, etc.
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

## Environment
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
- torchvision (0.1.8).
