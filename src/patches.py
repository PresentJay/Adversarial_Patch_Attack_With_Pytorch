import torch
import numpy as np

from utils import imgUtil

class AdversarialPatch():
    def __init__(self, dataset, target, device):
        self.dataset = dataset
        self.shape = dataset.shape[-2:]
        self.adversarial_image = None
        self.device = device
        self.mask = self.init_Mask()
        self.target = target
    
    def SetSquarePatch(self):
        pass
    
    def SetCirclePatch(self):
        pass
    
    def init_Patch(self):
        mean = torch.tensor(self.dataset.mean, dtype=torch.float)
        std = torch.tensor(self.dataset.std, dtype=torch.float)
        val = lambda x: ((x - mean) / std).to(self.device).unsqueeze(1).unsqueeze(1)
        self.val = {'min': val(0), 'max': val(1)}
        
        mask = torch.zeros(self.dataset.shape).to(self.device)
        
        
        
        mask = mask * (self.val['max'] - self.val['min']) + self.val['min']
        
        imgUtil.show_tensor(mask, block=True)
        
        min_index = np.argmin(self.shape)
        print(min_index)
        
        start = (self.shape[min_index-1] - self.shape[min_index])/2
        print(start)
        
        return mask