# UTSRMorph: Unified Transformer and Super-resolution Network for UnsupervisedMedical Image Registration.
Here is the official implementation of the paper.

## Update progress
4/24/2024 - UTSRMorph trained in OASIS datasets with dice loss is improved and the model trained in IXI datasets is publicy available!
4/15/2024 - UTSRMorph trained in OASIS datasets is now publicly available!

## Requirments
We trained our models depending on Pytorch 1.13.1 and Python 3.8.

## Train and infer
Our experiments are tested on 3 datasets: OASIS, IXI and our own dataset.
If you want to train OASIS dataset, you only need to run the following script: `train_UTSRMorph_oasis.py`. After the training stage, the model will be saved in `experients` folder.
To infer the trained model, you just need to run `infer_UTSRMorph.py` script.
The rest 2 datasets are the same as OASIS, the only difference is the path of dataset.

## Datasets
We offer 3 datasets: OASIS, IXI and our own dataset.
You can download the OASIS dataset from [OASIS]().
The IXI dataset can be downloaded from [IXI]().
Our dataset is avaiable on [Google Drive]().

## Contact
If you have any questions, feel free to contact zhangrunshi@buaa.edu.cn
