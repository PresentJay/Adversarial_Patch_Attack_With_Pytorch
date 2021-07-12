import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os, json

import numpy as np


class DataSet():
    def __init__(self, train, val, name, shape):
        self.name = name
        self.shape = shape
        if self.shape[1] == 224:  # when input image shape is 224x224 something
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            # TODO: concerns another model that uses not 224 size input
            pass
            
        # expandable area <---
        
        # ---/>
            
        self.set_trainset_ByFolder(train)
        self.set_valset_ByFolder(val)
        

    def Prepare(self):
        return transforms.Compose(
            [
                # https://github.com/pytorch/examples/issues/478
                # transforms.Resize() resizes the smallest edge to the given value. Thus, if the image is not square (height != width) Resize(224) would fail to give you an image of size (224, 224).
                transforms.Resize(self.shape[1] + 32),
                transforms.CenterCrop(self.shape[1]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
        )
        
        
    def set_trainset_ByFolder(self, source):
        self.trainset = ImageFolder(root=source, transform=self.Prepare())
    
    
    def set_valset_ByFolder(self, source):
        self.valset = ImageFolder(root=source, transform=self.Prepare())
        
        
    def SetDataLoader(self, batch_size, num_workers, pin_memory=True, shuffle=True):
        # pin_memory setting is good for GPU environments!
        # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        
        # Dataloader returns mini-batch data
        # if shuffle is True, dataloader shuffles all data every epoch. (it's opposite to sampler)
        
        self.train_loader = DataLoader(
            dataset=self.trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )
        
        self.val_loader = DataLoader(
            dataset=self.valset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )
        
    
    
    def GetTrainData(self):
        return self.train_loader
    
    
    def GetValData(self):
        return self.val_loader
    
    

def GetInfoFromLabel_ImageNet():
    with open('./data/ImageNetLabel.json') as f:
        json_obj = json.load(f)
    return json_obj


def GetWORDFromLabel_ImageNet(Label_tensor, ImageNet):
    Label_tensor = Label_tensor.item()
    return ImageNet[Label_tensor]['words']