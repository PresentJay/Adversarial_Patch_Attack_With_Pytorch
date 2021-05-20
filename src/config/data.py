# TODO: apply Google style Python Docstring

""" 
define about dataset
"""

from torchvision import datasets
from src.config.validator import NoneValidation

class Datasets:
    def __init__(self, dirName=None, size=0):
        self.size = size
        self.dir = f'../../data/{dirName}'
        NoneValidation(dirName=dirName)
    
class TrainSets(Datasets):
    def __init__(self, dirName=None, size=0):
        super(TrainSets, self).__init__('train', size)
        NoneValidation(dirName=dirName)
        self.dir += f'/{dirName}'
        
class ValSets(Datasets):
    def __init__(self, dirName=None, size=0):
        super(ValSets, self).__init__('val', size)
        NoneValidation(dirName=dirName)
        self.dir += f'/{dirName}'
        
class TestSets(Datasets):
    def __init__(self, dirName=None, size=0):
        super(TestSets, self).__init__('test', size)
        NoneValidation(dirName=dirName)
        self.dir += f'/{dirName}'
        
class Annotations(Datasets):
    def __init__(self, target=None, dirName=None, size=0):
        super(Annotations, self).__init__('ann', size)
        NoneValidation(dirName=dirName)
        self.dir += f'/{dirName}'
        NoneValidation(target=target)
        self.target = target
        