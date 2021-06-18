import torch
import numpy as np

from torchvision import transforms
from utils import imgUtil

class AdversarialPatch():
    def __init__(self, dataset, target, device, _type, hideProgress, random_init):
        self.dataset = dataset
        self.device = device
        self.hideProgress = hideProgress
        self._type = _type
        self.target = target
                
        mean = torch.tensor(self.dataset.mean, dtype=torch.float)
        std = torch.tensor(self.dataset.std, dtype=torch.float)
        val = lambda x: ((x - mean) / std).to(self.device).unsqueeze(1).unsqueeze(1)
        self.val = {'min': val(0), 'max': val(1)}
        
        self.patch = self.init_patch(random_init)

    
    def init_patch(self, random_init):
        if random_init:
            patch = torch.randn(self.dataset.shape).to(self.device)
            patch = patch * (self.val['max'] - self.val['min']) + self.val['min']
        else:
            patch = torch.zeros(self.dataset.shape).to(self.device)
        return patch
    
    
    def show(self):
        imgUtil.show_tensor(self.patch, block=True)
    
    
    def init_mask(self):        
        width = self.adversarial_image.shape[1]
        height = self.adversarial_image.shape[2]
        
        min_index = np.argmin(self.shape)
        print(min_index)
        
        start = (self.shape[min_index-1] - self.shape[min_index])/2
        print(start)
        
        # make shape to circle
        if _type == 'circle':
            pass
        
    
    
    # train patch for a one epoch
    def train(self, model, dataloader, target):
        model.eval()
        
        success = total = 0
        
        for batch_index, (data, labels) in enumerate(dataloader):
            batch_data = data.to(self.device)
            batch_labels = labels.to(self.device)
            
            imgUtil.show_tensor(self.Transformation(), block=True)
            
            
    def apply(self, image):
        pass
    
    
    def Transformation(self):        
        INTERPOLATION = transforms.InterpolationMode.BILINEAR
        
        rotation = transforms.RandomRotation(degrees=45)
        scale = transforms.RandomAffine(degrees=0, scale=(0.05, 0.3), interpolation=INTERPOLATION)
        location = transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), interpolation=INTERPOLATION)
        brightness = transforms.ColorJitter(brightness=(.4, 1.75))

        return transforms.Compose([
            scale,
            rotation,
            brightness,
            location
        ])\
            (self.patch)
