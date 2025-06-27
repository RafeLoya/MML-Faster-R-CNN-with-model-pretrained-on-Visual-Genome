#!/usr/bin/env python3
"""
Setup script to download and save PyTorch pretrained models
Run this script to create the pretrained model files that the faster-rcnn code expects.

Usage:
    python setup_pretrained_models.py
"""

import os
import torch
import torchvision.models as models


def download_and_save_pretrained_models():
    """Download PyTorch pretrained models and save them in the expected format."""

    # Create the pretrained model directory
    pretrained_dir = 'data/pretrained_model'
    os.makedirs(pretrained_dir, exist_ok=True)

    print("Setting up PyTorch pretrained models...")

    # Download and save VGG16
    print("Downloading VGG16 pretrained model...")
    vgg16 = models.vgg16(pretrained=True)
    vgg16_path = os.path.join(pretrained_dir, 'vgg16_pytorch.pth')
    torch.save(vgg16.state_dict(), vgg16_path)
    print(f"VGG16 model saved to: {vgg16_path}")

    # Download and save ResNet101
    print("Downloading ResNet101 pretrained model...")
    resnet101 = models.resnet101(pretrained=True)
    resnet101_path = os.path.join(pretrained_dir, 'resnet101_pytorch.pth')
    torch.save(resnet101.state_dict(), resnet101_path)
    print(f"ResNet101 model saved to: {resnet101_path}")

    # Optional: Download other ResNet variants
    print("Downloading additional ResNet models...")

    resnet50 = models.resnet50(pretrained=True)
    resnet50_path = os.path.join(pretrained_dir, 'resnet50_pytorch.pth')
    torch.save(resnet50.state_dict(), resnet50_path)
    print(f"ResNet50 model saved to: {resnet50_path}")

    resnet152 = models.resnet152(pretrained=True)
    resnet152_path = os.path.join(pretrained_dir, 'resnet152_pytorch.pth')
    torch.save(resnet152.state_dict(), resnet152_path)
    print(f"ResNet152 model saved to: {resnet152_path}")

    print("\nAll pretrained models have been downloaded and saved!")
    print("The modified faster-rcnn code will now work with PyTorch pretrained models.")

    # Print file sizes for reference
    print("\nDownloaded model file sizes:")
    for filename in os.listdir(pretrained_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(pretrained_dir, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename}: {size_mb:.1f} MB")


def verify_models():
    """Verify that the downloaded models can be loaded correctly."""
    print("\nVerifying downloaded models...")

    pretrained_dir = 'data/pretrained_model'
    models_to_check = [
        ('vgg16_pytorch.pth', models.vgg16),
        ('resnet101_pytorch.pth', models.resnet101),
        ('resnet50_pytorch.pth', models.resnet50),
        ('resnet152_pytorch.pth', models.resnet152)
    ]

    for model_file, model_class in models_to_check:
        model_path = os.path.join(pretrained_dir, model_file)
        if os.path.exists(model_path):
            try:
                # Create model and load weights
                model = model_class(pretrained=False)
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                print(f"  ✓ {model_file} - loaded successfully")
            except Exception as e:
                print(f"  ✗ {model_file} - failed to load: {e}")
        else:
            print(f"  ? {model_file} - file not found")


if __name__ == "__main__":
    download_and_save_pretrained_models()
    verify_models()

    print("\nNext steps:")
    print("1. Replace lib/model/faster_rcnn/resnet.py with the modified version")
    print("2. Replace lib/model/faster_rcnn/vgg16.py with the modified version")
    print("3. You can now use the faster-rcnn code with PyTorch pretrained models!")
    print("\nNote: The modified code will automatically fallback to PyTorch pretrained")
    print("models if the Caffe models are not found.")