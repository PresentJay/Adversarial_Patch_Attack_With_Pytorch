import argparse
import datetime
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import src.models as models

def Initialization():
    parser = argparse.ArgumentParser()
    
    # about Environments of this experiment
    parser.add_argument('--seed', '--s', metavar='SEED', type=int, default=None, help='Select the seed')
    parser.add_argument('--log-type', ttpe=str, default='txt', choices=['md', 'txt'])
    parser.add_argument('--num_workers', '--jobs', '--j', metavar='JOBS', type=int, default=4, help="num_workers (recommend to be half of your CPU cores)")
    parser.add_argument('--batch_size', type=int, default=20)
    
    # about Model
    parser.add_argument('--use-torchvision', '--use-tv', action='store_false')
    parser.add_argument('--model', '-m', metavar='MODEL', default='vgg16', choices=models.get_model_names(), help=f'choose model: {"|".join(models.get_model_names())} *defalt: vgg19')
    
    # about Dataset
    parser.add_argument('--data_dir', '--data', metavar='D', default='D:\datasets\ImageNet', help="dir of the dataset")
    parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network (basically 224, inception_v3(299) is not supported)')
    
    # about Training Adversarial Patch
    parser.add_argument('--iter', type=int, default=100000, help="it will served by batch size (must be divisable to batch_size)")

    # about Adversarial Patch Condition
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--target-class', '--target', '--t', metavar='T', type=int, default=859, help="The target class of adversarial patch : index 859 == toaster")
    parser.add_argument('--min-rot', type=float, default='-22.5')
    parser.add_argument('--max-rot', type=float, default='22.5')
    parser.add_argument('--min-scale', type=float, default='0.1')
    parser.add_argument('--max-scale', type=float, default='0.5')
    
    # about logging
    parser.add_argument('--result-dir', '--output-dir', metavar='DEST', default=str(datetime.datetime.now().date()), help='folder to output images and model checkpoints')
 
    args = parser.parse_args()
    
    assert not args.model.startswith('inception'), "inception series doesn't supported yet. . ."
    assert args.iter % args.batch_size == 0, "iteration size must be divisable to batch size"
        
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    args.imageshape = [3, args.image_size, args.image_size]
    
    return args


def init_directories(directoryName):
    try:
        os.makedirs(f'results/{directoryName}/candidate', exist_ok=True)
        os.makedirs(f'results/{directoryName}/best', exist_ok=True)    
    
    except OSError:
        pass
    