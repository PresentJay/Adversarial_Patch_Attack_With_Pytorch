"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

# TODO: apply Google style Python Docstring

import pickle
import traceback
from src import configs, datasets, models, patches
from utils import imgUtil, logUtil


if __name__ == '__main__':
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
        
    try:    
        # train adversarial patch
        patch = patches.AdversarialPatch(dataset=DataSet, target=args.target_class, device=args.device, random_init=args.random_init)
        log.write(f'start train patches with {args.iter_train} data iterations of trainset', _print=True)
        patch.train(classifier=classifier, iteration=args.iter_train, eot_dict=EOT_FACTORS, savedir=args.resultdir)
    except Exception as e:
        log.write(f'error occured during training patch: {traceback.format_exc()}', _print=True)
        if patch.fullyTrained:
            with open(args.resultdir + "/patch.pkl", "wb") as f:
                pickle.dump(patch.data.cpu(), f)
        else:
            log.write(f"a patch isn't fully trained. try again later.", _print=True)
        log.save()
    
    try:
        log.write(f'start validate patches with {args.iter_val} data iterations of validation set', _print=True)
        patch.measure_attackCapability(classifier=classifier, eot_dict=EOT_FACTORS, iteration=args.iter_val)
        log.write(f'Randomly patched model {classifier.getName()} has {patch.attackedAccuracy:.2f}% of accuracy to original, ', end='', _print=True)
        log.write(f'and {patch.attackCapability:.2f}% of accuracy to target {args.target_class}', _print=True)
        
        log.save()
        with open(args.resultdir + "/patch.pkl", "wb") as f:
            pickle.dump(patch.data.cpu(), f)
    except Exception as e:
        log.write(f'error occured during validating')
        with open(args.resultdir + "/patch.pkl", "wb") as f:
            pickle.dump(patch.data.cpu(), f)
        log.save()
        