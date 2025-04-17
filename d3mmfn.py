import logging
import os
import pickle
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import math

# PointNet++ utilities
from pointnet import *
from pointnet_util import *
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

LOGGER = logging.getLogger(__name__)

def init_weight(m):
    """Initialize weights using Xavier normal."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

# Discriminator: Detects normal vs anomalous features
class Discriminator(nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x

# Feature Adapter: Adapts the extracted features for the anomaly detection task
class FeatureAdapter(nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(FeatureAdapter, self).__init__()
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        x = self.layers(x)
        return x

# Fusion Layer: Combines features from both modalities (image + point cloud)
class FusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=1024):
        super(FusionLayer, self).__init__()
        self.fusion_fc = nn.Linear(input_dim, output_dim)
        self.apply(init_weight)

    def forward(self, image_features, point_cloud_features):
        # Concatenate image and point cloud features
        combined_features = torch.cat((image_features, point_cloud_features), dim=1)
        return F.relu(self.fusion_fc(combined_features))

# Anomalous Feature Generator: Adds Gaussian noise to normal features to simulate anomalies
class AnomalousFeatureGenerator(nn.Module):
    def __init__(self, noise_std=0.05):
        super(AnomalousFeatureGenerator, self).__init__()
        self.noise_std = noise_std

    def forward(self, normal_features):
        # Add Gaussian noise to normal features to simulate anomalies
        noise = torch.normal(0, self.noise_std, normal_features.shape).to(normal_features.device)
        anomalous_features = normal_features + noise
        return anomalous_features

# Main D3MMFN Model: Multimodal network that processes both images and point clouds
class D3MMFN(nn.Module):
    def __init__(self, device):
        super(D3MMFN, self).__init__()
        self.device = device
        
        # Feature Extractor for RGB Image (ResNet50 backbone)
        self.image_backbone = models.resnet50(pretrained=True)
        self.image_fc = nn.Linear(2048, 1024)  # ResNet50 output size is 2048

        # Feature Extractor for Point Cloud (PointNet++ backbone)
        self.point_cloud_encoder = encoder_BN_2(num_class=128)  # Example for PointNet++

        # Feature Adapter to adjust extracted features
        self.feature_adapter = FeatureAdapter(in_planes=1024 + 128, out_planes=1024, n_layers=2)
        
        # Fusion Layer to combine both image and point cloud features
        self.fusion_layer = FusionLayer(input_dim=1024 + 128, output_dim=1024)
        
        # Anomalous Feature Generator to simulate anomalies in feature space
        self.anomaly_feature_generator = AnomalousFeatureGenerator(noise_std=0.05)
        
        # Anomaly Discriminator to classify features as normal or anomalous
        self.anomaly_discriminator = Discriminator(in_planes=1024)

    def forward(self, rgb_image, point_cloud):
        # Process RGB Image
        image_features = self.image_backbone(rgb_image)
        image_features = F.relu(self.image_fc(image_features))

        # Process Point Cloud
        point_cloud_features = self.point_cloud_encoder(point_cloud)

        # Fusion of both modalities
        combined_features = self.fusion_layer(image_features, point_cloud_features)

        # Feature adaptation and anomaly generation
        adapted_features = self.feature_adapter(combined_features)
        anomalous_features = self.anomaly_feature_generator(adapted_features)

        # Anomaly detection
        anomaly_scores = self.anomaly_discriminator(anomalous_features)

        return anomaly_scores

    def embed(self, images, point_clouds):
        """Get embeddings for both image and point cloud modalities."""
        image_features = self.image_backbone(images)
        point_cloud_features = self.point_cloud_encoder(point_clouds)

        # Fusion of both modalities
        combined_features = self.fusion_layer(image_features, point_cloud_features)

        # Adapt features and generate anomaly features
        adapted_features = self.feature_adapter(combined_features)
        anomalous_features = self.anomaly_feature_generator(adapted_features)

        return anomalous_features


