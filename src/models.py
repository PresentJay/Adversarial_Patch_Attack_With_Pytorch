from torchvision import models as tvmodels
import torch

# this class enable you to experiments ensembled models or so.
class ModelContainer():
    def __init__(self):
        self.tvmodels = []
    
    def add_model(self, model):
        self.tvmodels.append(model)
    
    def get_model(self, index):
        return self.tvmodels[index]
    

# this class contains all variables about network models
class Model():
    def __init__(self, name, dataset, device, isTorchvision=True, explain=True):
        self.name = name
        self.explain = explain
        self.dataset = dataset
        if isTorchvision:
            self.model = self.load_model_from_torchvision()
            self.model.to(device)
        else:
            # TODO: setting for another models
            self.model = None
            
        if self.explain:
            print(f'Model {self.name} is loaded.\n. . . input shape is {self.dataset.shape}.')
        
    
    def load_model_from_torchvision(self):
        assert callable(tvmodels.__dict__[self.name]), 'undefined modelname in TorchVision'
        
        model = tvmodels.__dict__[self.name](pretrained=True)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
            
        return model


    def test(self, dataloader, device):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for index, (image, label) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.NetClassifier.model(images)
                
                # rank 1
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if total % 500 == 0 and self.explain:
                    accuracy = correct / total * 100
                    print(f'{total} of images are tested . . . intermediate score = {accuracy}')
                    
            if self.explain:
                print(f'{total} of images are tested complete . . . result score = {accuracy}')
                
        return accuracy


    def predict_once(self, image):
        output = self.NetClassifier.model(image)
        _, predicted = torch.max(output.data, 1)
        prediction = predicted[0].data.cpu().numpy()
        
        if self.explain:
            print(f'test image - - - {prediction} : {self.target}')
            print('success\n' if prediction == self.target else 'failed\n')
            return 1 if prediction == self.target else 0