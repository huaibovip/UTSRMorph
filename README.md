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
We offer 4 datasets: OASIS, Abdominal MR-CT, IXI and our own dataset.
You can download the Abdominial MR-CT dataset from [Abdominial MR-CT](https://drive.google.com/file/d/1R6bapU2UuAtmUTOrTJxMyDq-KxX8uZCo/view?usp=drive_link), the afterprocessed dataset can be downloaded from [Abdominial MR-CT](https://drive.google.com/file/d/1StPmkMCHKdM3a-yJQh8-bW6n6RkmaM92/view?usp=drive_link).
The IXI and OASIS dataset can be downloaded from [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration).
Our dataset is avaiable on [Google Drive]().

## Contact
If you have any questions, feel free to contact zhangrunshi@buaa.edu.cn

## Reference and Acknowledgments
[TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)

[Swin Transformer](https://github.com/microsoft/Swin-Transformer)

[VoxelMorph](https://github.com/voxelmorph/voxelmorph)

[TransMatch](https://github.com/tzayuan/TransMatch_TMI)
