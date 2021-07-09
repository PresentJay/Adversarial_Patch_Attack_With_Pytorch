import torch
from utils import imgUtil
from src.datasets import GetInfoFromLabel_ImageNet, GetWORDFromLabel_ImageNet
from torchvision import models as tvmodels

# this class enable you to experiments ensembled models or so.
class ModelContainer():
    def __init__(self):
        self.tvmodels = []
    
    def add_model(self, model):
        self.tvmodels.append(model)
    
    def get_model(self, index):
        return self.tvmodels[index]
    
    # don't use. . .!
    def test_models(self, original=False):
        for model in self.tvmodels:
            if original:
                model.test(original)
            else:
                model.test()
                
    def get_models(self):
        return self.tvmodels
    

# this class contains all variables about network models
class Model():
    def __init__(self, name, device, isTorchvision=True):
        self.name = name
        self.device = device
        self.scores = []
        if isTorchvision:
            self.model = self.load_model_from_torchvision()
        else:
            # TODO: setting for another models
            self.model = None
            
        if not self.hideProgress:
            print(f'\nModel {self.name} is loaded.\n. . .')
        
    
    def load_model_from_torchvision(self):
        assert callable(tvmodels.__dict__[self.name]), 'undefined modelname in TorchVision'
        
        model = tvmodels.__dict__[self.name](pretrained=True)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        if self.name.startswith('alexnet') or self.name.startswith('vgg'):
            model.to(self.device)
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model).to(self.device)
            
        return model


    def test(self, dataloader, original=False):
        correct = 0
        total = 0
        accuracy = 0.0
        
        with torch.no_grad():
            if not self.hideProgress:
                print(f'\nstart test {self.name} model. . .')
            
            imgnet = GetInfoFromLabel_ImageNet()
            
            for index, (images, labels) in enumerate(dataloader):
                # imgUtil.show_tensor(images=images, title='original', text=labels.item())
                # imgUtil.show_batch_data(images=images, labels=labels, block=True)
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                
                # rank 1
                _, predicted = torch.max(outputs, 1)
                
                # for idx, (i, l, p) in enumerate(zip(images, labels, predicted)):
                #     print(f'{index} batch / {idx} : label {GetWORDFromLabel_ImageNet(l, imgnet)} : predicted {GetWORDFromLabel_ImageNet(p, imgnet)}  correct? <{l==p}>')
                    # imgUtil.show_tensor(images=i, title=GetWORDFromLabel_ImageNet(l, imgnet), text=GetWORDFromLabel_ImageNet(p, imgnet), block=True)  
                
                # imgUtil.show_tensor(images=images, title='prediction', text=predicted.item(), block=True)
                total += labels.size(0)  # concern batch_size
                correct += (predicted == labels).sum().item()
                    
            accuracy = correct / total * 100
            if not self.hideProgress:
                print(f'{total} of images are tested complete . . . result score = {accuracy}')
        
        if original:
            self.original_score = accuracy
        else:
            self.scores.append(accuracy)
            

    def predict(self, image):
        self.model.eval()
        
        image.to(self.device)
        output = self.model(image)
    
        probability, predicted = torch.max(output, 1)
        if len(predicted)==1:
            return probability, predicted.item()
        return probability, predicted
        

def get_model_names():
    names = []
    for name in tvmodels.__dict__:
        if name.islower() and \
            not name.startswith('__') and\
                callable(tvmodels.__dict__[name]):
            names.append(name)
    return sorted(names)

