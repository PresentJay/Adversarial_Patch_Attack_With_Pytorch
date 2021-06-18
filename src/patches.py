import torch
import numpy as np

from utils import imgUtil

class AdversarialPatch():
    def __init__(self, dataset, target, device, _type, hideProgress):
        self.dataset = dataset
        self.device = device
        self.hideProgress = hideProgress
        self._type = _type
        self.target = target
        
        # self.shape = imgUtil.reducing_rectangle(dataset.shape, reduce_rate=reduce_rate)
        
        mean = torch.tensor(self.dataset.mean, dtype=torch.float)
        std = torch.tensor(self.dataset.std, dtype=torch.float)
        val = lambda x: ((x - mean) / std).to(self.device).unsqueeze(1).unsqueeze(1)
        self.val = {'min': val(0), 'max': val(1)}
        
        self.adversarial_image = self.init_Adversarial_Image()

        # self.mask = self.init()
    
    
    
    def init_Adversarial_Image(self, random_init=False):
        if random_init:
            adversarial_image = torch.rand(self.dataset.shape).to(self.device)
        else:
            adversarial_image = torch.zeros(self.dataset.shape).to(self.device)
        
        # rand val[min] to val[max]
        adversarial_image = adversarial_image * (self.val['max'] - self.val['min']) + self.val['min']
        
        if not self.hideProgress:
            imgUtil.show_tensor(adversarial_image, block=True)
            print(adversarial_image.is_cuda)
        
        return adversarial_image
    
    
    def clamp_patch_to_valid(self, patch):
        ch_ranges = [
            [-self.dataset.mean[0] / self.dataset.std[0], (1 - self.dataset.mean[0]) / self.dataset.std[0]],
            [-self.dataset.mean[1] / self.dataset.std[1], (1 - self.dataset.mean[1]) / self.dataset.std[1]],
            [-self.dataset.mean[2] / self.dataset.std[2], (1 - self.dataset.mean[2]) / self.dataset.std[2]]
        ]
        with torch.no_grad():
            self.patch[0] = torch.clamp(self.patch[0], ch_ranges[0][0], ch_ranges[0][1])
            self.patch[1] = torch.clamp(self.patch[1], ch_ranges[1][0], ch_ranges[1][1])
            self.patch[2] = torch.clamp(self.patch[2], ch_ranges[2][0], ch_ranges[2][1])

    
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
        
        return mask
    
    
    # train patch
    def train(self, model, ):
        model.eval()
        
        success = total = 0