import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import transforms
from utils import imgUtil

class AdversarialPatch():
    def __init__(self, dataset, target, device, random_init):
        self.dataset = dataset
        self.device = device
        self.target = target
                
        if random_init:
            self.patch = torch.randn(self.dataset.shape).to(self.device)
        else:
            self.patch = torch.zeros(self.dataset.shape).to(self.device)
                
        mean = torch.tensor(self.dataset.mean, dtype=torch.float)
        std = torch.tensor(self.dataset.std, dtype=torch.float)
        val = lambda x: ((x - mean) / std).to(self.device).unsqueeze(1).unsqueeze(1)
        self.val = {'min': val(0), 'max': val(1)}
        self.patch.requires_grad = True
    
    
    def show(self):
        imgUtil.show_tensor(self.patch, block=True)
        
        
    def clamp(self):
        with torch.no_grad():
            ch_ranges = [
                    [-self.dataset.mean[0] / self.dataset.std[0], (1 - self.dataset.mean[0]) / self.dataset.std[0]],
                    [-self.dataset.mean[1] / self.dataset.std[1], (1 - self.dataset.mean[1]) / self.dataset.std[1]],
                    [-self.dataset.mean[2] / self.dataset.std[2], (1 - self.dataset.mean[2]) / self.dataset.std[2]],
            ]

            self.patch[0] = torch.clamp(self.patch[0], ch_ranges[0][0], ch_ranges[0][1])
            self.patch[1] = torch.clamp(self.patch[1], ch_ranges[1][0], ch_ranges[1][1])
            self.patch[2] = torch.clamp(self.patch[2], ch_ranges[2][0], ch_ranges[2][1])
        
    
    # train patch for a one epoch
    def train(self, classifier, iteration, eot_dict, savedir, log):
        train_size = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        finish_trigger = True
        while finish_trigger:
            for batch_index, (images, labels) in enumerate(self.dataset.GetTrainData()):
                train_size += images.size(0)
                batch_images = images.to(self.device)
                target_vector = torch.tensor([target]).repeat(batch_images.shape[0]).to(self.device)
                                
                self.patch.detach_()
                self.patch.requires_grad = True
                
                eot_variables = self.set_transform_variables(size=images.size(0), eot_dict=eot_dict)
                patched_images = self.attach(batch_images, eot_variables)
                output = classifier.model(patched_images)
                
                loss = criterion(output, target_vector)
                loss.backward()
                
                self.patch.data = self.patch.data - self.patch.grad.data
                self.clamp()
                
                if train_size % (iteration / 5) == 0:
                    pil_image = imgUtil.tensor_to_PIL(self.patch, self.dataset.mean, self.dataset.std)
                    print(f"patch trained {train_size} iteration . . .", end='')
                    try:
                        pil_image.save(f"{savedir}/patch{train_size}.png")
                        print("saved.")
                    except Exception as e:
                        print(f"can't be saved => cause {e}")
                
                if train_size >= iteration:
                    finish_trigger = False
                    break
                
                
    def attach(self, data, eot_variables):
        circle_mask = torch.tensor(imgUtil.to_circle(), dtype=torch.float).to(self.device)
        masks = circle_mask.repeat(data.size(0), 1, 1, 1)
        patches = self.patch.repeat(data.size(0), 1, 1, 1)
        
        
        pass
                
        
    def set_transform_variables(self, size, eot_dict):
        # [0] : scale
        # [1] : rotation
        # [2] : locationX
        # [3] : locationY
        
        eot_variables=[]
        
        locationX = torch.empty(size)
        locationY = torch.empty(size)
        if eot_dict.has_key('scale'):
            eot_variables.append(
                torch.empty(size).uniform_(eot_dict['scale'][0], eot_dict['scale'][1])
            )
        if eot_dict.has_key('rotation'):
            eot_variables.append(
                torch.empty(size).uniform_(eot_dict['rotation'][0], eot_dict['rotation'][1])
            )
        for i in range(size):
            _scale = eot_variables[0][i]
            locationX[i] = np.random.uniform(_scale - 1, 1 - _scale)
            locationY[i] = np.random.uniform(_scale - 1, 1 - _scale)
        eot_variables.append(locationX)
        eot_variables.append(locationY)
                    
        return eot_variables