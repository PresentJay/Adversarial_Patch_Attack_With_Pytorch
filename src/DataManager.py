""" 
extract samples from dataset if it is too large and user wants.

using import torchvision.transforms
"""

from torchvision import transforms

def DataLoader(trainSize, dataset, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop()
    ])
        