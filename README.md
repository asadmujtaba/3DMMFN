
# 3DMMFN

## Introduction

This repository contains the official source code for **3D-MMFN: Multi-level Multimodal Fusion Network for 3D Industrial Image Anomaly Detection**.

## Requirement

- Ubuntu 18.04

## Environment

- Python >= 3.8

## Packages

- torch
- torchvision
- numpy
- opencv-python

Other required package(s) can be installed via:

### Install pointnet2_ops_lib
```bash
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## RUN Model

For **RGB Only model**:
```bash
bash run_RgbOModel.sh
```

For **3DMMFN model**:
```bash
bash run_3dmmfn.sh
```

## Data Download and Preprocess

### Dataset

The **MVTec-3D AD dataset** can be downloaded from the [Official Website of MVTec-3D AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).

The **Eyecandies dataset** can be downloaded from the [Official Website of Eyecandies](https://eyecan-ai.github.io/eyecandies/).

After downloading, move the dataset to the `data` folder.

### Data Preprocess

To run the preprocessing for background removal and resizing, run the following command:

```bash
python preprocess/preprocessing_mvtec3d.py
```

## Citation

If you find this repository useful for your research, please use the following citation:

```bibtex
@article{asad20253d,
  title={3D-MMFN: Multi-level multimodal fusion network for 3D industrial image anomaly detection},
  author={Asad, Mujtaba and Azeem, Waqar and Malik, Aftab Ahmad and Jiang, He and Ali, Ahmad and Yang, Jie and Liu, Wei},
  journal={Advanced Engineering Informatics},
  volume={65},
  pages={103284},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement

This repo is based on **SimpleNet** ([link](https://github.com/DonaldRR/SimpleNet)) and **M3DM** ([link](https://github.com/nomewang/M3DM)), and we thank them for their great work.
