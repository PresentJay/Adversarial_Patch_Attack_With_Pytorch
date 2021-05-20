# TODO: apply Google style Python Docstring

from torchvision import models

RESNET50 = 'resnet50'
CUSTOM = 'custom'
      

def load_model(name=CUSTOM, device=None):
    if name == CUSTOM:
        # TODO: try to handle the custom models
        pass
    else:
        assert callable(models.__dict__[name]), 'undefined modelname yet. please read our doc.'
        model = models.__dict__[name](pretrained=True)
        model.eval()
        return model
    