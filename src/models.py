from torchvision import models as tvmodels
import torch

# this class enable you to experiments ensembled models or so.
class ModelContainer():
    def __init__(self, device):
        self.tvmodels = []
        self.device = device
    
    def add_model(self, model):
        self.tvmodels.append(model)
    
    def get_model(self, index):
        return self.tvmodels[index]
    

# this class contains all variables about network models
class Model():
    def __init__(self, name, dataset, isTorchvision=True, explain=True):
        self.name = name
        self.explain = explain
        self.dataset = dataset
        if isTorchvision:
            load_model_from_torchvision()
        else:
            # TODO: setting for another models
            self.NetClassifier = None
            
        if self.explain:
            print(f'Model {self.name} is loaded.\ninput shape is {self.dataset.shape}.\nmean&std is {self.dataset.mean}, {self.dataset.std}.')
        
    
    def load_model_from_torchvision(self):
        assert self.isTorchVision, "you didn't set torchvision model but it is in <load_model_from_torchvision>_function."
        assert callable(tvmodels.__dict__[self.name]), 'undefined modelname in TorchVision'
        
        NetClassifier = tvmodels.__dict__[self.name](pretrained=True)
        NetClassifier.eval()
        
        for param in NetClassifier.parameters():
            param.requires_grad = False
            
        if self.name.startswith('inception'):
            self.inputshape = [3, 299, 299]
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            self.inputshape = [3, 244, 244]
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            
        self.NetClassifier = NetClassifier


    def test(self, dataloader):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for index, (image, label) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.NetClassifier.model(images)
                
                # rank 1
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if total % 500 == 0 and self.explain:
                    accuracy = correct / total * 100
                    print(f'{total} of images are tested . . . intermediate score = {accuracy}')
        
        return accuracy


    def predict_once(self, image):
        output = self.NetClassifier.model(image)
        _, predicted = torch.max(output.data, 1)
        prediction = predicted[0].data.cpu().numpy()
        
        if self.explain:
            print(f'test image - - - {prediction} : {self.target}')
            print('success\n' if prediction == self.target else 'failed\n')
            return 1 if prediction == self.target else 0