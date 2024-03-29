import argparse
import datetime
import torch
import random
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
plt.ioff()

from src.models import get_model_names

def Initialization():
    parser = argparse.ArgumentParser()
    
    # about Environments of this experiment
    parser.add_argument('--seed', '--s', metavar='SEED', type=int, default=None)
    parser.add_argument('--log-type', type=str, default='txt', choices=['md', 'txt'])
    parser.add_argument('--num-workers', '--jobs', '--j', metavar='JOBS', type=int, default=4,
                        help="num_workers (recommend to be half of your CPU cores)")
    parser.add_argument('--batch-size', type=int, default=20)
    
    # TODO: if this network be implemented to another models not using imagenet, choices will be thrown out!
    # about Model
    parser.add_argument('--model', '-m', metavar='MODEL', default='vgg16', choices=get_model_names(),
                        help=f'choose model: {"|".join(get_model_names())} *defalt: vgg16')
    parser.add_argument('--torchvision', action='store_false')
    
    # about Dataset
    parser.add_argument('--train-dir', default=os.path.join('D:', 'datasets', 'imagenet', 'val'))
    parser.add_argument('--val-dir', default=os.path.join('D:', 'datasets', 'imagenet', 'train'))
    
    # about Training Adversarial Patch
    parser.add_argument('--iter-train', type=int, default=100000,
                        help="it will served by batch size (must be divisable to batch_size)")
    parser.add_argument('--iter-val', type=int, default=5000,
                        help="it will served by batch size (must be divisable to batch_size)")

    # about Adversarial Patch Condition
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--target-class', '--target', '--t', metavar='T', type=int, default=859,
                        help="The target class of adversarial patch : index 859 == toaster")
    parser.add_argument('--min-rotation', type=float, default='-22.5')
    parser.add_argument('--max-rotation', type=float, default='22.5')
    parser.add_argument('--min-scale', type=float, default='0.1')
    parser.add_argument('--max-scale', type=float, default='0.5')
 
 
    # parsing arguments
    args = parser.parse_args()
    
    
    # taking early-exceptions   
    assert not args.model.startswith('inception'), "inception series doesn't supported yet. . ."
    assert args.iter_train % args.batch_size == 0, "train iteration size must be divisable to batch size"
    assert args.iter_val % args.batch_size == 0, "val iteration size must be divisable to batch size"
    
    
    if args.model.startswith('inception'):
        args.imageshape = [3, 299, 299]
    else:
        args.imageshape = [3, 224, 224]
    
        
    # initiating seed for randomizing
    random.seed(args.seed)
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    np.random.seed(args.seed)
    
    
    
    # setting devices
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        torch.manual_seed(args.seed)
        
    
    # initiating directories of result
    scale_label = f"x({args.min_scale*100},{args.max_scale*100})"
    rot_label = f"rot({args.min_rotation},{args.max_rotation})"
    condition_label = f"{args.model}_to_{args.target_class}_{args.iter_train}iter_{scale_label}_{rot_label}"
    timed = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    args.resultdir = os.path.join('results', condition_label, timed)
    try:
        os.makedirs(f'{args.resultdir}', exist_ok=True)
    except OSError:
        pass
    
    
    return args