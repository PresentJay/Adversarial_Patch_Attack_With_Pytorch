import torchvision
from torchvision import models

# TODO: apply Google style Python Docstring

def LoadModel(name, pretrained=True):
    model = getattr(models, name).cuda()
    model.eval()
    
    