"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

# TODO: apply Google style Python Docstring

import torch

from src.config import models, datasets, DataManager

if __name__ == '__main__':
    
    # 1. set available process units alternative CPU and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. load model
    model = models.load_model(models.RESNET50, device=device)
    
    # 3. load datasets
    dataset = datasets.Datasets(dirName="d:/datasets/imagenet1k", otherDir=True)
    
    # 4. configure DataManager
    train_loader, test_loader = DataManager