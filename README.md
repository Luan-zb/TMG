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
ADI|10407
BACK|10566
DEB|11512
LYM|11557
MUC|8896
MUS|13536
NORM|8763
STR|10446
TUM|14317

## Environment
- Python (3.7.7)
- h5py (2.10.0)
- matplotlib (3.1.1)
- numpy (1.18.1)
- opencv-python (4.1.1)
- openslide-python (1.1.1)
- openslide (3.4.1)
- pandas (1.0.3)
- pillow (7.0.0)
- PyTorch (1.5.1)
- scikit-learn (0.22.1)
- scipy (1.3.1)
- tensorflow (1.14.0)
- tensorboardx (1.9)
- torchvision (0.6).
