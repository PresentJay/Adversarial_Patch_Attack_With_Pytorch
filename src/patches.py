import torch
import numpy as np

from utils import imgUtil

class AdversarialPatch():
    def __init__(self, dataset, target, device, _type, explain):
        self.dataset = dataset
        self.device = device
        self.explain = explain
        self._type = _type
        self.target = target
        
        # self.shape = imgUtil.reducing_rectangle(dataset.shape, reduce_rate=reduce_rate)
        
        mean = torch.tensor(self.dataset.mean, dtype=torch.float)
        std = torch.tensor(self.dataset.std, dtype=torch.float)
        val = lambda x: ((x - mean) / std).to(self.device).unsqueeze(1).unsqueeze(1)
        self.val = {'min': val(0), 'max': val(1)}
        
        self.adversarial_image = self.init_Adversarial_Image()

        # self.mask = self.init()
    
    
    
    def init_Adversarial_Image(self, random=True):
        if random:
            adversarial_image = torch.rand(self.dataset.shape).to(self.device)
        else:
            adversarial_image = torch.zeros(self.dataset.shape).to(self.device)
        
        # rand val[min] to val[max]
        adversarial_image = adversarial_image * (self.val['max'] - self.val['min']) + self.val['min']
        
        if self.explain:
            imgUtil.show_tensor(adversarial_image, block=True)
            print(adversarial_image.is_cuda)
        
        return adversarial_image

    
    def init_mask(self):        
        
        min_index = np.argmin(self.shape)
        print(min_index)
        
        start = (self.shape[min_index-1] - self.shape[min_index])/2
        print(start)
        
        # make shape to circle
        if _type == 'circle':
            pass
        
        return mask
    
    
    # train patch
    def train(self, model):
        model.eval()
        
        success = total = 0
        
        