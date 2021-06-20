import torch
import torch.nn.functional as F
from torch.autograd import Variable
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
        # transformed_patches = []
        # transformed_masks = []
        # factors = []
        
        # for i in range(data.shape[0]):
        #     transformed_patch, transformed_mask, factor = self.Transformation()
        #     transformed_patches.append(transformed_patch)
        #     transformed_masks.append(transformed_mask)
        #     factors.append(factor)
            
        
        # stack을 통해 batch 단위로 patch작업!
        # transformed_patches = torch.stack(transformed_patches, dim=0)
        # transformed_masks = torch.stack(transformed_masks, dim=0)
        
        transformed_patch, transformed_mask, factor = self.Transformation()
        
        # clamp 안 하면 patch 안 붙여짐!
        masked_image = torch.clamp(data - transformed_mask, 0, 255)
        
        # patch 붙임!
        patched_image = masked_image + transformed_patch
        
        return patched_image, factor
    
    
    # train patch for a one epoch
    def train(self, model, dataloader, target, lr, prob_threshold, max_iteration):
        success = 0
        total = 0
        # success = total = 0 하면 공유되나..? !TODO: 알아보기
        torch.set_grad_enabled(True)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([self.patch], lr=lr)
        
        for batch_index, (data, labels) in enumerate(dataloader):
            total += data.shape[0]
            batch_data = data.to(self.device)
            batch_labels = labels.to(self.device)
            
            optimizer.zero_grad()
            model.model.zero_grad()
            
            patched_data, factor = self.attach(batch_data)
            output = model.model(patched_data)
            target_probability = F.softmax(output, dim=1)[0][target].item()
            if target_probability > prob_threshold:
                continue
            loss = criterion(output, torch.tensor(data.shape[0] * [target]).to(self.device))
            
            loss.backward()
            optimizer.step()
            
            patched_data, factor = self.attach(batch_data)
            new_output = model.model(patched_data)
            new_target_probability = F.softmax(output, dim=1)[0][target].item()
            print(f'batch {batch_index} : attacked prob={target_probability:.2f}% >> {new_target_probability:.2f}%')
            
            # _, original_predict = model.predict(batch_data)

            # # 정확한 분류를 한 케이스 :1
            # correct_candidate = (batch_labels == original_predict).type(torch.IntTensor)
            # # 분류 결과가 target이 아닌 케이스 :2
            # predict_candidate = (original_predict != self.target).type(torch.IntTensor)
            # # 원래 레이블이 target이 아닌 케이스 :3
            # label_candidate = (batch_labels != self.target).type(torch.IntTensor)
            
            # # 1, 2, 3을 모두 만족하는 경우 candidate로 설정
            # candidate_index = correct_candidate + predict_candidate + label_candidate
            # candidate = batch_data[candidate_index == 3]
            
            # # add candidate to total size
            # total += candidate.shape[0]
            
            # # candidate에 대해 patch attach 수행
            # patched_candidate, factors = self.attach(candidate)
            
            
            # # patched_candidate가 존재하는 경우 학습 페이즈
            # if patched_candidate.shape[0] > 0:
            #     output = model.model(patched_candidate)
            #     target_probability = F.softmax(output, dim=1)[0][target].item()
            #     print(f'batch {batch_index} : start prob={target_probability:.2f}%')
                
            #     target_tensor = torch.tensor([target]).repeat(patched_candidate.shape[0]).to(self.device)
            #     iteration = 0
                
            #     while target_probability < prob_threshold:
                    
            #         iteration += 1
            #         patched_variable = Variable(patched_candidate.data, requires_grad=True)
            #         output = model.model(patched_variable)
            #         logit = F.log_softmax(output)
            #         loss = -logit[0][target]
            #         # loss = criterion(output, target_tensor)
            #         loss.backward()
                    
            #         patched_grad = patched_variable.grad.clone()
            #         patched_variable.grad.data.zero_()
                    
            #         for i in range(patched_grad.shape[0]):
            #             # temp = self.patch
            #             # print("loss:", patched_grad[i])
            #             self.patch = self.patch - lr * patched_grad[i]
            #             # print("sub:", self.patch - temp)
                        
            #         self.patch = torch.clamp(self.patch, min=0, max=255)
                    
            #         # delete GPU cache data
            #         torch.cuda.empty_cache()
                    
            #         patched_candidate, factors = self.attach(candidate)
            #         output = model.model(patched_candidate)
            #         target_probability = F.softmax(output, dim=1)[0][target].item()
            #         print(f'batch {batch_index} : {iteration} attacked prob={target_probability:.2f}%')

            #         if iteration==max_iteration:
            #             break
                
                
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
        patch = transforms.functional.affine(img=patch, angle=0, scale=1.0, translate=location, shear=[0, 0], interpolation=INTERPOLATION).to(self.device)
        mask = transforms.functional.affine(img=mask, angle=0, scale=1.0, translate=location, shear=[0, 0], interpolation=INTERPOLATION).to(self.device)

        return patch, mask, expectation_over_transformation_factors


