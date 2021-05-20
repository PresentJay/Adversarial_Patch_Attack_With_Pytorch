"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

# TODO: apply Google style Python Docstring

import torch

from src.config.models import *
from src.config.data import *

if __name__ == '__main__':
    
    # 1. set available process units alternative CPU and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. load model
    model = load_model(Models().resnet50, device=device)
    
    # 3. load datasets
    dataset = Annotations()