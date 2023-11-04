# Dynamic Neural Fields for Learning Atlases of 4D Fetal MRI Time-series

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/clintonjwang/sinf/blob/main/LICENSE)

This repository contains the PyTorch implementation of the paper **Dynamic Neural Fields for Learning Atlases of 4D Fetal MRI Time-series**, accepted by [Medical Imaging Meets NeurIPS 2023](https://sites.google.com/view/med-neurips2023).

![img](teaser.png)

## Environment Setup

Clone this repo:

```shell
git clone https://github.com/Kidrauh/neural-atlasing.git
```

### General Environment

If your Linux system is Ubuntu 22.04 with gcc 11.3, please execute the following command:

```shell
conda env create -f environment.yml
```

The environment we provide uses CUDA 11.7, which is compatible with gcc 11.3. If you're using lower versions of gcc, please use CUDA 11.3 or any compatible version of CUDA, but the runtime may be stretched.

Some key packages:

- Python 3.8
- PyTorch 1.13
- [Tinycudann](https://github.com/NVlabs/tiny-cuda-nn)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- OpenCV
- scikit-learn
- nibabel
- scipy
- wandb

When the packages are installed, run `pip install -e .` to install the code framework.

### WanDB Setup

We use [WanDB](https://wandb.ai/) for runtime status inspection. You should register for an account if you don't have one, and save your own WanDB API key locally in `.wandb` file. Please explicitly set the environment variable `$NFS` in your own computer and organize your directory structure like this:

```
|-- $NFS
   |-- .wandb
   |-- code
      |-- sinf
         |-- ...
```

## Dataset Structure

Our fetal BOLD MRI dataset is private, but we offer the structure of the dataset below, so that as long as you organize your own fetal MRI dataset following our instruction, the code can run normally.

Please explicitly set the environment variable `$DS_DIR` in your own computer first. Suppose we have several fetal MRI subjects, named after subject1, subject2, etc., and each subject contains a sequence of fetal MRI nifti files and their corresponding segmentation files (either placental or multi-label segmentation). For atlas-as-brigde evaluation, suppose the indices of images or segmentations of each pair are stored in `pairs.txt`, where each row of `pairs.txt` only contains two indices separated by a space. You should organize your dataset as below:

```
|-- $DS_DIR
   |-- subject1
   |  |-- images
   |  |  |-- img1.nii.gz
   |  |  |-- img2.nii.gz
   |  |  |-- ...
   |  |-- segs
   |  |  |-- seg1.nii.gz
   |  |  |-- seg2.nii.gz
   |  |  |-- ...
   |  |-- pairs.txt
   |-- subject2
   |  |-- images
   |  |  |-- ...
   |  |-- segs
   |  |  |-- ...
   |  |-- pairs.txt
   |-- ...
```

## Training
