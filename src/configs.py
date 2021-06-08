import argparse
import time
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def Initialization():
    parser = argparse.ArgumentParser()
    
    # about Environments of this experiment
    parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')
    parser.add_argument('--showProgress', action='store_true', default=True, help='show process logs for your understanding')
    parser.add_argument('--num_workers', type=int, default=4, help="num_workers (to be half of your CPU cores")
    
    # about Dataset
    parser.add_argument('--trainsize', type=int, default=2000, help="number of training data")
    parser.add_argument('--testsize', type=int, default=400, help="number of testing data")
    
    parser.add_argument('--data_dir', type=str, default='D:\datasets\ImageNet', help="dir of the dataset")
    parser.add_argument('--image_size', type=int, default=244, help='the height / width of the input image to network (basically 244, inception_v3 is 299')
    parser.add_argument('--batch_size', type=int, default=10, help="batch size")
    
    # about Training Adversarial Patch
    parser.add_argument('--noise_percentage', type=float, default=0.1, help="percentage of the patch size compared with the image size")
    parser.add_argument('--max_iteration', type=int, default=1000, help="max number of iterations to find adversarial example")
    parser.add_argument('--patch_size', type=float, default=0.5, help='patch size. E.g. 0.05 ~= 5% of image ')
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
    parser.add_argument('--epochs', type=int, default=20, help="total epoch")
    parser.add_argument('--momentum', type=float, default=0.9, help="set momentum parameter (in SGD optimizer)")

    # about Adversarial Patch Condition
    parser.add_argument('--target', type=int, default=859, help="The target class index: 859 == toaster")
    parser.add_argument('--patch_type', type=str, default='circle', help="type of the patch => (circle or rectangle)")
    parser.add_argument('--probability_threshold', type=float, default=0.9, help="Stop attack on image when target classifier reaches this value for target class")
    
    # about logging
    parser.add_argument('--logdir', default='./logs', help='folder to output images and model checkpoints')
 
    args = parser.parse_args()
        
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    
    args.imageshape = [3, args.image_size, args.image_size]
    
    return args


def init_directories(directoryName):
    try:
        os.makedirs(f'results/{directoryName}/candidate', exist_ok=True)
        os.makedirs(f'results/{directoryName}/best', exist_ok=True)
        
    
    except OSError:
        pass
    