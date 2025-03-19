# 3DMMFN
Official source code for "3D-MMFN: Multi-level Multimodal Fusion Network for 3D Industrial Image Anomaly Detection"

## Project Structure
# 3D-MMFN/
# ├── preprocess_normals.py   # Compute & save surface normal maps
# ├── dataset.py              # Data loading pipeline
# ├── models/
# │   ├── __init__.py         # Init file
# │   ├── pointnet2.py        # PointNet++ for 3D features
# │   ├── resnet.py           # ResNet backbone for 2D features
# │   ├── fusion.py           # Multimodal feature fusion
# │   ├── anomaly_detector.py # Discriminator for anomaly detection
# ├── train.py                # Training script
# ├── evaluate.py             # Model evaluation
# ├── utils.py                # Visualization and helper functions
# ├── requirements.txt        # Dependencies
# ├── run.sh                  # Shell script to run training

