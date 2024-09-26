# UTSRMorph: A Unified Transformer and Superresolution Network for Unsupervised Medical Image Registration. (TMI2024)
<strong><big>Keywords:</big></strong> Deformable image registration, ConvNets, Transformer, Cross-attention, Superresolution.

Here is the <strong><big>PyTorch implementation</big></strong> of the paper:

[R. Zhang et al., "UTSRMorph: A Unified Transformer and Superresolution Network for Unsupervised Medical Image Registration," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2024.3467919.](https://ieeexplore.ieee.org/document/10693635))

## Update progress
23/9/2024 - The [paper](https://ieeexplore.ieee.org/document/10693635) is accepted in <strong><big>IEEE TMI</big></strong>.

31/8/2024 - UTSRMorph trained in Abdominal MR-CT and CMF tumor MR-CT datasets is now publicly available!

4/24/2024 - UTSRMorph trained in OASIS datasets with dice loss is improved and the model trained in IXI datasets is publicy available!

4/15/2024 - UTSRMorph trained in OASIS datasets is now publicly available!

## Requirments
We trained our models depending on Pytorch 1.13.1 and Python 3.8.

## Train and infer
UTSRMorph are tested on 4 datasets: OASIS, IXI, Abdominal MR-CT and CMF tumor MR-CT datasets.
If you want to train OASIS dataset, you only need to run the following script: `train_UTSRMorph_oasis.py`. After the training stage, the model will be saved in `experients` folder.
To infer the trained model, you just need to run `infer_UTSRMorph.py` script.
The rest 3 datasets are the same as OASIS, the only difference is the path of dataset.

## Datasets
4 datasets: OASIS, IXI, Abdominal MR-CT and CMF tumor MR-CT dataset.
The IXI and OASIS dataset can be downloaded from [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration).
You can download the Abdominial MR-CT dataset from [Abdominial MR-CT](https://drive.google.com/file/d/1R6bapU2UuAtmUTOrTJxMyDq-KxX8uZCo/view?usp=drive_link), the afterprocessed dataset can be downloaded from [Abdominial MR-CT](https://drive.google.com/file/d/1StPmkMCHKdM3a-yJQh8-bW6n6RkmaM92/view?usp=drive_link).
The CMF tumor MR-CT dataset is avaiable on [Google Drive](https://drive.google.com/file/d/1Ugi_C_0JdyAxuzYT55t_Be-hAn-iNuLK/view?usp=drive_link).

## Contact
If you have any questions, feel free to contact zhangrunshi@buaa.edu.cn

## Reference and Acknowledgments
[TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)

[Swin Transformer](https://github.com/microsoft/Swin-Transformer)

[VoxelMorph](https://github.com/voxelmorph/voxelmorph)

[TransMatch](https://github.com/tzayuan/TransMatch_TMI)
