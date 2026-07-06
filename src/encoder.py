"""
VisionEncoder — CPU-Optimized Feature Extraction using ResNet-18.

This module provides a memory-safe vision backbone that strips the
classification head from a pre-trained ResNet-18 and outputs a raw
512-dimensional feature vector. Designed for edge deployment on
CPU-only devices with strict RAM constraints.

CRITICAL MEMORY CONSTRAINTS:
    - Model is explicitly mapped to CPU via map_location=torch.device('cpu')
    - Forward pass runs under torch.no_grad() to prevent gradient allocation
    - Model is permanently locked in .eval() mode to disable batch-norm tracking
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class VisionEncoder:
    """
    Extracts 512-dimensional feature vectors from images using a headless
    ResNet-18 backbone. All operations are CPU-bound and gradient-free.
    """

    def __init__(self):
        """
        Initialize the ResNet-18 backbone:
        1. Load pre-trained weights, explicitly mapped to CPU.
        2. Remove the final fully-connected (fc) classification layer.
        3. Lock the model in eval() mode permanently.
        """
        # Load pre-trained ResNet-18 with explicit CPU mapping
        self.device = torch.device('cpu')
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Strip the final classification layer — replace with Identity
        # This causes the model to output the 512-D feature vector from avgpool
        self.model.fc = nn.Identity()

        # CRITICAL: Lock to eval mode to disable dropout & batch-norm tracking
        self.model.eval()

        # Ensure all parameters are on CPU
        self.model = self.model.to(self.device)

        # Standard ImageNet preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract a normalized 512-dimensional feature vector from an image.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            np.ndarray: A 1-D NumPy array of shape (512,) containing the
                        L2-normalized feature embedding.

        Memory Safety:
            The entire forward pass executes under torch.no_grad() to
            prevent PyTorch from allocating memory for gradient computation.
            This is non-negotiable on edge devices with ≤8GB RAM.
        """
        # Load image via Pillow and convert to RGB (handles grayscale/RGBA)
        image = Image.open(image_path).convert('RGB')

        # Apply ImageNet preprocessing and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # CRITICAL: No-grad context prevents OOM from gradient accumulation
        with torch.no_grad():
            features = self.model(input_tensor)

        # Flatten to 1-D and convert to NumPy
        feature_vector = features.squeeze().cpu().numpy()

        # L2 normalize the feature vector for stable downstream distance metrics
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm

        return feature_vector
