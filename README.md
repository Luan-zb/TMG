# TN-MIL-GAP
 tissue_classification
Identify various tissues in a pathological section, including but not limited to stroma, lymphocytes, tumor cells, etc.
## Data
Source data from https://zenodo.org/record/1214456#.YotWsKhBy5c.

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
- bitarray: 2.5.1
- pylibtiff: 0.4.4 (please install bitarray before pylibtiff)
