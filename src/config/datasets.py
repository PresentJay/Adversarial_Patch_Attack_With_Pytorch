# TODO: apply Google style Python Docstring

""" 
define about datasets
"""

from torchvision import datasets
from src.config.validator import NoneValidation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class Datasets:
    def __init__(self, dirName=None, size=0, otherDir=False):
        self.size = size
        NoneValidation(dirName=dirName)
        if otherDir:
            NoneValidation(dirName=dirName)
            self.dir = dirName
        else:
            self.dir = f'../../data/{dirName}'
    
class TrainSets(Datasets):
    def __init__(self, dirName=None, size=0, otherDir=False):
        if otherDir:
            NoneValidation(dirName=dirName)
            self.dir = dirName
        else:
            super(TrainSets, self).__init__(dirName='train', size=size)
            NoneValidation(dirName=dirName)
            self.dir += f'/{dirName}'
        
class ValSets(Datasets):
    def __init__(self, dirName=None, size=0, otherDir=False):
        if otherDir:
            NoneValidation(dirName=dirName)
            self.dir = dirName
        else:
            super(ValSets, self).__init__(dirName='val', size=size)
            NoneValidation(dirName=dirName)
            self.dir += f'/{dirName}'
        
class TestSets(Datasets):
    def __init__(self, dirName=None, size=0, otherDir=False):
        if otherDir:
            NoneValidation(dirName=dirName)
            self.dir = dirName
        else:
            super(TestSets, self).__init__(dirName='test', size=size)
            NoneValidation(dirName=dirName)
            self.dir += f'/{dirName}'
        
class Annotations(Datasets):
    def __init__(self, target=None, dirName=None, size=0, otherDir=False):
        if otherDir:
            NoneValidation(dirName=dirName)
            self.dir = dirName
        else:
            super(Annotations, self).__init__(dirName='ann', size=size)
            NoneValidation(dirName=dirName)
            self.dir += f'/{dirName}'
            NoneValidation(target=target)
            self.target = target
