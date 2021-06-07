import numpy as np
import torch
import torch.cuda
import torch.nn
from typing import Optional, Union
from torch.types import Number
from utils.images import create_circular_mask, transform
from torchvision import transforms


class AdversarialPatch():
    
    def __init__(self, image=None, mask=None, target=None):
        self.image = image
        self.mask = mask
        self.adv_image = None
        self.target = None
        self._shape = None
        