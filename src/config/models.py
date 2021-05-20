# TODO: apply Google style Python Docstring

from torchvision import models

class Models:
    def __init__(self):
      self.custom = 'custom'
      self.resnet50 = 'resnet50'
      

def load_model(name=Models().custom, device=None):
    if name == Models().custom:
        # TODO: try to handle the custom models
        pass
    else:
        assert callable(models.__dict__[name]), 'undefined modelname yet. please read our doc.'
        model = models.__dict__[name](pretrained=True)
        model.eval()
        return model
    