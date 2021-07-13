import torch
import math
import traceback
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils import imgUtil, accUtil

class AdversarialPatch():
    def __init__(self, dataset, target, device, random_init):
        self.dataset = dataset
        self.device = device
        self.target = target
                
        if random_init:
            self.patch = torch.randn(self.dataset.shape).to(self.device)
        else:
            self.patch = torch.zeros(self.dataset.shape).to(self.device)

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
    def train(self, classifier, iteration, eot_dict, savedir):
        train_size = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        finish_trigger = True
        running_state = 0
        while finish_trigger:
            for batch_index, (images, labels) in enumerate(self.dataset.GetTrainData()):
                train_size += images.size(0)
                batch_images = images.to(self.device)
                target_vector = torch.tensor([self.target]).repeat(batch_images.shape[0]).to(self.device)
                                
                self.patch.detach_()
                self.patch.requires_grad = True
                
                eot_variables = self.set_transform_variables(size=images.size(0), eot_dict=eot_dict)
                patched_images = self.attach(batch_images, eot_variables)
                output = classifier.model(patched_images)
                
                loss = criterion(output, target_vector)
                loss.backward()
                
                self.patch.data = self.patch.data - self.patch.grad.data
                self.clamp()
                if train_size % (iteration / 100) == 0:
                    running_state += 1
                    print(f'a patch is trained by {train_size} iteration . . . ({running_state}%)')
                
                if train_size % (iteration / 5) == 0:
                    pil_image = imgUtil.tensor_to_PIL(self.patch, self.dataset.mean, self.dataset.std)
                    print(f"a patch is trained by {train_size} iteration . . .", end='')
                    try:
                        pil_image.save(f"{savedir}/patch{train_size}.png")
                        print("saved.")
                    except Exception as e:
                        print(f"can't be saved => cause {traceback.format_exc()}")
                
                if train_size >= iteration:
                    finish_trigger = False
                    break
        torch.cuda.empty_cache()
                
                
    def attach(self, images, eot_variables):
        circle_mask = torch.tensor(imgUtil.to_circle(images.shape[1:]), dtype=torch.float).to(self.device)
        masks = circle_mask.repeat(images.size(0), 1, 1, 1)
        patches = self.patch.repeat(images.size(0), 1, 1, 1)
        
        coefficients = self.coefficients_for_transformation(*eot_variables)
        transform_grid = F.affine_grid(coefficients, images.size(), align_corners=True).to(self.device)
        
        transformed_mask = F.grid_sample(masks, transform_grid, align_corners=True)
        transformed_patch = F.grid_sample(patches, transform_grid, align_corners=True)
        
        return images * (1 - transformed_mask) + transformed_patch * transformed_mask
                
        
    def set_transform_variables(self, size, eot_dict):
        assert 'scale' in eot_dict, "must contain the scale range"
        assert 'rotation' in eot_dict, "must contain the rotation range"
        
        locationX = torch.empty(size)
        locationY = torch.empty(size)
        scale = torch.empty(size).uniform_(eot_dict['scale'][0], eot_dict['scale'][1])
        rotation = torch.empty(size).uniform_(eot_dict['rotation'][0], eot_dict['rotation'][1])
        for i in range(size):
            _scale = scale[i]
            locationX[i] = np.random.uniform(_scale - 1, 1 - _scale)
            locationY[i] = np.random.uniform(_scale - 1, 1 - _scale)
                    
        return scale, rotation, locationX, locationY
    
    
    def coefficients_for_transformation(self, scale, rotation, locationX, locationY):
        rot = rotation / 90. * (math.pi/2)
        
        cos = torch.cos(-rot)
        sin = torch.sin(-rot)
        
        cos /= scale
        sin /= scale
        locationX /= -scale
        locationY /= -scale
        
        return torch.stack((cos, -sin, locationX, sin, cos, locationY)).t().view(-1, 2, 3)
    
    
    def measure_attackCapability(self, classifier, eot_dict, iteration):
        acc = accUtil.Accuracy()
        attack_capability = accUtil.Accuracy()
        
        for index, (images, labels) in enumerate(self.dataset.GetValData()):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            eot_variables = self.set_transform_variables(size=images.size(0), eot_dict=eot_dict)
            patched_images = self.attach(images, eot_variables)
            output = classifier.model(patched_images)
            
            target_vector = torch.tensor([self.target]).repeat(output.size(0)).to(self.device)
            corrects = acc.calculate(output, labels)
            attacked = attack_capability.calculate(output, target_vector)
            if ((index+1) * images.size(0)) >= iteration:
                break
        
        self.attackedAccuracy = acc.average()
        self.attackCapability = attack_capability.average()