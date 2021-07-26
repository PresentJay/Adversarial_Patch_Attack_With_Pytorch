# TODO: apply Google style Python Docstring
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import traceback
from src import configs, datasets, models, patches
from utils import imgUtil, logUtil

def initializer():
    # load the Network Settings
    args = configs.Initialization()
    
    log = logUtil.log(args.resultdir, args.log_type)
    log.write(f'Random Seed: {args.seed}', _print=True)
    log.write(f'rotation: {args.min_rotation}-{args.max_rotation} degree', _print=True)
    log.write(f'scale: {args.min_scale*100}%-{args.max_scale*100}%', _print=True)
    EOT_FACTORS = {
        "scale" : (args.min_scale, args.max_scale),
        "rotation": (args.min_rotation, args.max_rotation)
    }
    
    try:
        # set the models
        classifier = models.Model(name=args.model, device=args.device, isTorchvision=args.torchvision)
        
        # load the dataset
        DataSet = datasets.DataSet(train=args.train_dir, val=args.val_dir, name="ImageNet", shape=args.imageshape)
        DataSet.SetDataLoader(batch_size=args.batch_size, num_workers=args.num_workers)
        log.write(f'train dataloader: {len(DataSet.trainset)}', end='', _print=True)
        print(f' are ready from {args.train_dir}', end='')
        log.write(f'\nvalidate dataloader: {len(DataSet.valset)}', end='', _print=True)
        print(f' are ready from {args.val_dir}', end='')
        log.write(f'\nbatch size: {args.batch_size}', _print=True)
        # measure label-accuracy and attack_capability of classifier
        classifier.measure_attackCapability(dataloader=DataSet.GetValData(), iteration=args.iter_val, target=args.target_class)
        log.write(f'loaded model {classifier.getName()} has {classifier.getAccuracy():.2f}% of accuracy to original,', end=' ', _print=True)
        log.write(f'and {classifier.getAttackCapability():.2f}% of accuracy to target {args.target_class}', _print=True)
    except Exception as e:
        log.write(f'error occured before create patch: {traceback.format_exc()}', _print=True)
        log.save()