# TODO: apply Google style Python Docstring
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import traceback
from src import configs, datasets, models, patches
from utils import imgUtil, logUtil

def initializer(logname='log', logmode='w'):
    # load the Network Settings
    args = configs.Initialization()
    
    log = logUtil.log(savedir=args.resultdir, ext=args.log_type, mode=logmode, name=logname)
    log.write(f'Random Seed: {args.seed}', _print=True)
    log.write(f'rotation: {args.min_rotation}-{args.max_rotation} degree', _print=True)
    log.write(f'scale: {args.min_scale*100}%-{args.max_scale*100}%', _print=True)
    
    try:
        # set the models
        classifier = models.Model(name=args.model, device=args.device, isTorchvision=args.torchvision)
        
        # load the dataset
        dataset = datasets.DataSet(train=args.train_dir, val=args.val_dir, name="ImageNet", shape=args.imageshape)
        dataset.SetDataLoader(batch_size=args.batch_size, num_workers=args.num_workers)
        log.write(f'train dataloader: {len(dataset.trainset)}', end='', _print=True)
        print(f' are ready from {args.train_dir}', end='')
        log.write(f'\nvalidate dataloader: {len(dataset.valset)}', end='', _print=True)
        print(f' are ready from {args.val_dir}', end='')
        log.write(f'\nbatch size: {args.batch_size}', _print=True)
        # measure label-accuracy and attack_capability of classifier
        classifier.measure_attackCapability(dataloader=dataset.GetValData(), iteration=args.iter_val, target=args.target_class)
        log.write(f'loaded model {classifier.getName()} has {classifier.getAccuracy():.2f}% of accuracy to original,', end=' ', _print=True)
        log.write(f'and {classifier.getAttackCapability():.2f}% of accuracy to target {args.target_class}', _print=True)
    except Exception as e:
        log.write(f'error occured before create patch: {traceback.format_exc()}', _print=True)
        log.save()
    
    return classifier, dataset, args, log