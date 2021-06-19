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
        self.mask = self.init_mask()
        
        
    def init_mask(self):
        mask = torch.full(self.dataset.shape, 255.).to(self.device)
        # mask = torch.ones(self.dataset.shape).to(self.device)
        return mask

    
    def init_patch(self, random_init):
        if random_init:
            patch = torch.randn(self.dataset.shape).to(self.device)
            patch = patch * (self.val['max'] - self.val['min']) + self.val['min']
        else:
            patch = torch.zeros(self.dataset.shape).to(self.device)
        return patch
    
    
    def show(self):
        imgUtil.show_tensor(self.patch, block=True)
    
    
    
    # train patch for a one epoch
    def train(self, model, dataloader, target):
        success = total = 0
        
        for batch_index, (data, labels) in enumerate(dataloader):
            batch_data = data.to(self.device)
            batch_labels = labels.to(self.device)
            
            # add batch_size to total size
            total += batch_data.shape[0]
            
            transformed_patch, transformed_mask, factors = self.Transformation()
            patched_image = self.apply(transformed_patch, transformed_mask, batch_data)
            predict = model.predict(patched_image)
            
            correct = batch_labels==self.target
            attacked = predict==self.target
            success += (correct!=attacked).sum()
            print(f'{batch_index} batch : attacked {attacked.sum()} / already target {correct.sum()} >> ({success}/{total})')
            
            # success += (predict == batch_labels)
            
            # if predict == self.target:
            #     success += (predict != batch_labels)
            #     print(f'{batch_index} batch : ({success}/{total})')
        
            
    def apply(self, transformed_patch, transformed_mask, image):
        masked_image = torch.clamp(image - transformed_mask,0,255)
        # imgUtil.show_tensor(image, title="image", block=False)
        # imgUtil.show_tensor(transformed_patch, title="patch", block=False)
        # imgUtil.show_tensor(masked_image, title="image - mask", block=False)
        patched_image = masked_image + transformed_patch
        # imgUtil.show_tensor(patched_image, title="image + patch", block=True)
        return patched_image
        
        
    def getRandomFactors(self):
        rotation = transforms.RandomRotation.get_params(degrees=(-45, 45))
        _, location, scale, __ = transforms.RandomAffine.get_params(
            degrees=(0, 0),
            translate=(0.3, 0.3),
            scale_ranges=(0.05, 0.3),
            shears=[0, 0],
            img_size=self.dataset.shape[1:]
        )
        brightness = transforms.ColorJitter.get_params(brightness=(.4, 1.75), contrast=None, saturation=None, hue=None)
        
        return rotation, location, scale, brightness
    
    
    def Transformation(self):
        # TODO: BILINEAR와 NEAREST 사이 성능 차이?
        INTERPOLATION = transforms.InterpolationMode.NEAREST
        
        rotation, location, scale, brightness = self.getRandomFactors()
        expectation_over_transformation_factors = {
            "rotation" : rotation,
            "location" : location,
            "scale" : scale,
            "brightness" : brightness[1]
        }
        
        # print(expectation_over_transformation_factors)
        
        # scale
        patch = transforms.functional.affine(img=self.patch, angle=0, scale=scale, translate=[0, 0], shear=[0, 0], interpolation=INTERPOLATION)
        mask = transforms.functional.affine(img=self.mask, angle=0, scale=scale, translate=[0, 0], shear=[0, 0], interpolation=INTERPOLATION)
        
        # rotation
        patch = transforms.functional.rotate(img=patch, angle=rotation, interpolation=INTERPOLATION)
        mask = transforms.functional.rotate(img=mask, angle=rotation, interpolation=INTERPOLATION)
        
        # brightness
        patch = transforms.functional.adjust_brightness(img=patch, brightness_factor=brightness[1])
        
        # location
        patch = transforms.functional.affine(img=patch, angle=0, scale=1.0, translate=location, shear=[0, 0], interpolation=INTERPOLATION)
        mask = transforms.functional.affine(img=mask, angle=0, scale=1.0, translate=location, shear=[0, 0], interpolation=INTERPOLATION)

        return patch, mask, expectation_over_transformation_factors
