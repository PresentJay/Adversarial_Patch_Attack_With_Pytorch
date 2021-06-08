import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os

import numpy as np


class DataSet():
    def __init__(self, source, name, shape, trainfull, trainsize, testfull, testsize, explain):
        self.name = name
        self.shape = shape
        self.explain = explain
        
        if self.shape[1] == 299:    # when input image shape is 299x299 something
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif self.shape[1] == 244:  # when input image shape is 244x244 something
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        # expandable area <---
        
        # ---/>
            
        # TODO: figure out the space to EOT variables.
        # self.transformed = kwargs.pop('transformed')
        
        self.train_index = self.Shuffle(fullsize=trainfull, size=trainsize)
        self.test_index = self.Shuffle(fullsize=testfull, size=testsize)
            
        self.LoadByFolder(source)
        # LoadByImageNet(source)
        
        if self.explain:
            print(f'dataset {self.name} is loaded from [{source}].')
            print(f'train data size is {trainsize}, test data size is {testsize}.')
            
    
    def Shuffle(self, fullsize, size):
        index = np.arange(fullsize)
        np.random.shuffle(index)
        return index[:size]
        

    def Prepare(self):
        # if self.transformed:
        
        return transforms.Compose(
            [
                # plain prepare
                transforms.Resize((self.shape[1], self.shape[2])),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )
        
        """ else:
            # prepare Expectation Over Transformation
            return transforms.Compose(
                [
                    # To prepare EOT variables area <---
                    
                    # ---/>
                    
                    # plain prepare
                    transforms.Resize((self.shape[1], self.shape[2])),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]
            ) """
            
        
    def LoadByFolder(self, source):
        self.trainset = ImageFolder(root=os.path.join(source, 'train'), transform=self.Prepare())
        self.testset = ImageFolder(root=os.path.join(source, 'val'), transform=self.Prepare())
    
        
    def LoadByImageNet(self, source):
        self.trainset = ImageNet(root=source, split="train", transform=self.Prepare())
        self.testset = ImageNet(root=source, split="val", transform=self.Prepare())
        
        
    def SetDataLoader(self, batch_size, num_workers, pin_memory=True, shuffle=False):
        # pin_memory setting is good for GPU environments!
        # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        
        self.train_loader = DataLoader(
            dataset=self.trainset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(self.train_index),
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )
        
        self.test_loader = DataLoader(
            dataset=self.testset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(self.test_index),
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )
        
        if self.explain:
            print('. . . dataloaders are ready.')
        
    
    
    def GetTrainData(self):
        return self.train_loader
    
    
    def GetTestData(self):
        return self.test_loader