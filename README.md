# 3DMMFN
Official source code for "3D-MMFN: Multi-level Multimodal Fusion Network for 3D Industrial Image Anomaly Detection"

3D-MMFN/
├── preprocess_normals.py       # Script to compute and save surface normal maps
├── dataset.py                  # Data loading pipeline for training and evaluation
├── models/
│   ├── __init__.py             # Initialization file for models package
│   ├── pointnet2.py            # PointNet++ model for extracting 3D features
│   ├── resnet.py               # ResNet backbone for extracting 2D features
│   ├── fusion.py               # Multimodal feature fusion module
│   ├── anomaly_detector.py     # Anomaly detection discriminator
├── train.py                    # Script to train the model
├── evaluate.py                 # Model evaluation script
├── utils.py                    # Utility functions (visualization, helpers, etc.)
├── requirements.txt            # Python dependencies
├── run.sh                      # Shell script to execute training process
