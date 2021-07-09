import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os, json

import numpy as np


class DataSet():
    def __init__(self, source, name, shape):
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
        
        self.train_index = self.Shuffle(fullsize=trainfull, size=trainsize)
        self.test_index = self.Shuffle(fullsize=testfull, size=testsize)
            
        self.LoadByFolder(source)
        # LoadByImageNet(source)
                    
    
    def Shuffle(self, fullsize, size):
        index = np.arange(fullsize)
        np.random.shuffle(index)
        return index[:size]
        

    def Prepare(self):
        return transforms.Compose(
            [
                # https://github.com/pytorch/examples/issues/478
                # transforms.Resize() resizes the smallest edge to the given value. Thus, if the image is not square (height != width) Resize(224) would fail to give you an image of size (224, 224).
                
                transforms.Resize(256),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(self.mean, self.std)
            ]
        )
        
        
    def LoadByFolder(self, source):
        self.trainset = ImageFolder(root=os.path.join(source, 'train'), transform=self.Prepare())
        self.testset = ImageFolder(root=os.path.join(source, 'val'), transform=self.Prepare())
    
        
    def LoadByImageNet(self, source):
        self.trainset = ImageNet(root=source, split="train", transform=self.Prepare())
        self.testset = ImageNet(root=source, split="val", transform=self.Prepare())
        
        
    def SetDataLoader(self, batch_size, num_workers, pin_memory=True, shuffle=False):
        # pin_memory setting is good for GPU environments!
        # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        
        # Dataloader returns mini-batch data
        # if shuffle is True, dataloader shuffles all data every epoch. (it's opposite to sampler)
        
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
        
        if not self.hideProgress:
            print('. . . dataloaders are ready.')
        
    
    
    def GetTrainData(self):
        return self.train_loader
    
    
    def GetTestData(self):
        return self.test_loader
    
    

def GetInfoFromLabel_ImageNet():
    with open('./data/ImageNetLabel.json') as f:
        json_obj = json.load(f)
    return json_obj


def GetWORDFromLabel_ImageNet(Label_tensor, ImageNet):
    Label_tensor = Label_tensor.item()
    return ImageNet[Label_tensor]['words']