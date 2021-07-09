import argparse
import datetime
import torch
import random
import os
import numpy as np

import matplotlib
matplotlib.use('agg')
matplotlib.pyplot.ioff()

from src.models import get_model_names

def Initialization():
    parser = argparse.ArgumentParser()
    
    # about Environments of this experiment
    parser.add_argument('--seed', '--s', metavar='SEED', type=int, default=None)
    parser.add_argument('--log-type', ttpe=str, default='txt', choices=['md', 'txt'])
    parser.add_argument('--num_workers', '--jobs', '--j', metavar='JOBS', type=int, default=4,
                        help="num_workers (recommend to be half of your CPU cores)")
    parser.add_argument('--batch_size', type=int, default=20)
    
    # TODO: if this network be implemented to another models not using imagenet, choices will be thrown out!
    # about Model
    parser.add_argument('--model', '-m', metavar='MODEL', default='vgg16', choices=get_model_names(),
                        help=f'choose model: {"|".join(get_model_names())} *defalt: vgg16')
    parser.add_argument('--torchvision', action='store_false')
    
    # about Dataset
    parser.add_argument('train-dir', default=os.path.join('D:', 'datasets', 'imagenet', 'val'))
    parser.add_argument('val-dir', default=os.path.join('D:', 'datasets', 'imagenet', 'train'))
    
    parser.add_argument('--image_size', type=int, default=224,
                        help='the height / width of the input image to network (basically 224, inception_v3(299) is not supported)')
    
    # about Training Adversarial Patch
    parser.add_argument('--iter-train', type=int, default=100000,
                        help="it will served by batch size (must be divisable to batch_size)")
    parser.add_argument('--iter-val', type=int, default=5000,
                        help="it will served by batch size (must be divisable to batch_size)")

    # about Adversarial Patch Condition
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--target-class', '--target', '--t', metavar='T', type=int, default=859,
                        help="The target class of adversarial patch : index 859 == toaster")
    parser.add_argument('--min-rot', type=float, default='-22.5')
    parser.add_argument('--max-rot', type=float, default='22.5')
    parser.add_argument('--min-scale', type=float, default='0.1')
    parser.add_argument('--max-scale', type=float, default='0.5')
 
 
    # parsing arguments
    args = parser.parse_args()
    
    
    # taking early-exceptions   
    assert not args.model.startswith('inception'), "inception series doesn't supported yet. . ."
    assert args.iter % args.batch_size == 0, "iteration size must be divisable to batch size"
        
        
    # initiating seed for randomizing
    random.seed(args.seed)
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    np.random.seed(args.seed)
    
    
    # setting image_shape
    args.imageshape = [3, args.image_size, args.image_size]
    
    
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
    rot_label = f"rot({args.min_angle},{args.max_angle})"
    condition_label = f"{args.arch}_to_{args.target_class}_{args.steps*args.batch_size}iter_{scale_label}_{rot_label}"
    timed = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    args.resultdir = os.path.join(args.output, condition_label, timed)
    try:
        os.makedirs(f'results/{args.resultdir}/progress_patches', exist_ok=True)
    except OSError:
        pass
    
    
    return args