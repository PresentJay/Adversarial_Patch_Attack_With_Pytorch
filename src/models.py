import torch

from utils import imgUtil, accUtil
# from src.datasets import GetInfoFromLabel_ImageNet, GetWORDFromLabel_ImageNet
from torchvision import models as tvmodels
    

# this class contains all variables about network models
class Model():
    def __init__(self, name, device, isTorchvision=True):
        self.name = name
        self.device = device
        if isTorchvision:
            self.model = self.load_model_from_torchvision()
            print(f"=> using pre-trained model '{self.name}'")
        else:
            # TODO: setting for another models
            print(f"=> creating model '{self.name}'")
            self.model = None
            
    
    def load_model_from_torchvision(self):
        assert callable(tvmodels.__dict__[self.name]), 'undefined modelname in TorchVision'
        
        model = tvmodels.__dict__[self.name](pretrained=True)
        model.eval()
        
        if self.name.startswith('alexnet') or self.name.startswith('vgg'):
            model.to(self.device)
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model).to(self.device)
        
        for param in model.parameters():
            param.requires_grad = False
            
        return model


    def validate(self, dataloader, _iter):
        acc = accUtil.Accuracy()
        
        for index, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            corrects = acc.calculate(outputs, labels)
            if ((index+1) * images.size(0)) >= _iter:
                break
        
        self.accuracy = acc.average()
        
        

    def measure_attackCapability(self, dataloader, iteration, target):
        acc = accUtil.Accuracy()
        attack_capability = accUtil.Accuracy()
        
        for index, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            target_vector = torch.tensor([target]).repeat(outputs.size(0)).to(self.device)
            corrects = acc.calculate(outputs, labels)
            attacked = attack_capability.calculate(outputs, target_vector)
            if ((index+1) * images.size(0)) >= iteration:
                break
        
        self.accuracy = acc.average()
        self.attackCapability = attack_capability.average()
        
        
    def getName(self):
        return self.name
    
    
    def getAccuracy(self):
        return self.accuracy
    
    
    def getAttackCapability(self):
        return self.attackCapability
    


def get_model_names():
    names = []
    for name in tvmodels.__dict__:
        if name.islower() and \
            not name.startswith('__') and\
                callable(tvmodels.__dict__[name]):
            names.append(name)
    return sorted(names)



# this class enable you to experiments ensembled models or so.
# class ModelContainer():
#     def __init__(self):
#         self.tvmodels = []
    
#     def add_model(self, model):
#         self.tvmodels.append(model)
    
#     def get_model(self, index):
#         return self.tvmodels[index]
    
#     # don't use. . .!
#     def test_models(self, original=False):
#         for model in self.tvmodels:
#             if original:
#                 model.test(original)
#             else:
#                 model.test()
                
#     def get_models(self):
#         return self.tvmodels