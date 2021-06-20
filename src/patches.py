import torch
import torch.nn.functional as F
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
        self.patch.requires_grad = True
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
        
        
    def attach(self, data):
        transformed_patches = []
        transformed_masks = []
        factors = []
        for i in range(data.shape[0]):
            transformed_patch, transformed_mask, factor = self.Transformation()
            transformed_patches.append(transformed_patch)
            transformed_masks.append(transformed_mask)
            factors.append(factor)
            
        # stack을 통해 batch 단위로 patch작업!
        transformed_patches = torch.stack(transformed_patches)
        transformed_masks = torch.stack(transformed_masks)
            
        # clamp 안 하면 patch 안 붙여짐!
        masked_image = torch.clamp(data - transformed_masks, 0, 255)
        
        # patch 붙임!
        patched_image = masked_image + transformed_patches
        
        return patched_image, factors
    
    
    # train patch for a one epoch
    def train(self, model, dataloader, target, lr):
        success = 0
        total = 0
        # success = total = 0 하면 공유되나..? !TODO: 알아보기
        
        lr = lr
        criterion = torch.nn.CrossEntropyLoss()
        
        for batch_index, (data, labels) in enumerate(dataloader):
            batch_data = data.to(self.device)
            batch_labels = labels.to(self.device)
            
            # add batch_size to total size
            total += batch_data.shape[0]

            # except incorrect predictions and except original label is target.
            _, original_predict = model.predict(batch_data)
              
            correct_candidate = (batch_labels == original_predict).type(torch.IntTensor)
            predict_candidate = (original_predict != self.target).type(torch.IntTensor)
            label_candidate = (batch_labels != self.target).type(torch.IntTensor)
            candidate_index = correct_candidate + predict_candidate + label_candidate
            candidate = batch_data[candidate_index == 3]
            candidate_labels = batch_labels[candidate_index == 3]
            
            patched_image, factors = self.attach(candidate)
            _, predict = model.predict(patched_image)
            
            # except attacked candidate
            patched_candidate = patched_image[predict!=self.target]
            
            if patched_candidate.shape[0] > 0:
                target_tensor = torch.tensor([target]).repeat(patched_candidate.shape[0]).to(self.device)
                
                self.patch.detach_()
                self.patch.requires_grad = True
                
                output = model.model(patched_candidate)
                loss = criterion(output, target_tensor)
                loss.backward()
                
                self.patch.data -= lr * self.patch.grad.data
                self.patch.data = torch.clamp(self.patch.data, 0, 255)
                
                self.show()
                
                # imgUtil.show_batch_data(patched_candidate, title="show_candidates", block=True)
                
                
            
            # correct = batch_labels==self.target
            # attacked = predict==self.target
            # success += (correct!=attacked).sum()
            # print(f'{batch_index} batch : attacked {attacked.sum()} / already target {correct.sum()} >> ({success}/{total})')
            
            # success += (predict == batch_labels)
            
            # if predict == self.target:
            #     success += (predict != batch_labels)
            #     print(f'{batch_index} batch : ({success}/{total})')
        
        
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


