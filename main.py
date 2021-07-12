"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

# TODO: apply Google style Python Docstring

import pickle
from src import configs, datasets, models, patches
from utils import imgUtil, logUtil


if __name__ == '__main__':
    # load the Network Settings
    args = configs.Initialization()
    
    log = logUtil.log(args.resultdir, args.log_type)
    log.write(f'Random Seed: {args.seed}', _print=True)
    log.write(f'rotation: {args.min_rot}-{args.max_rot} degree', _print=True)
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
        log.write(f'train dataloader: {len(DataSet.trainset)} are ready from {args.train_dir}', _print=True)
        log.write(f'validate dataloader: {len(DataSet.valset)} are ready from {args.val_dir}', _print=True)
        
        # measure label-accuracy and attack_capability of classifier
        classifier.measure_attackCapability(dataloader=DataSet.GetValData(), _iter=args.iter_val, target=args.target_class)
        log.write(f'loaded model {classifier.getName()} has {classifier.getAccuracy():.2f}% of accuracy to original,', end=' ', _print=True)
        log.write(f'and {classifier.getAttackCapability():.2f}% of accuracy to target {args.target_class}', _print=True)
    except Exception as e:
        log.write(f'error occured: {e}', _print=True)
        log.save()
        
    try:    
        # train adversarial patch
        patch = patches.AdversarialPatch(dataset=DataSet, target=args.target_class, device=args.device, random_init=args.random_init)
        log.write(f'train patches with {args.iter} data iterations of trainset', _print=True)
        patch.train(classifier=classifier, iteration=args.iter, eot_dict=EOT_FACTORS, savedir=args.resultdir, log=log)
        
        log.save()
        with open(args.resultdir + "/patch.pkl", "wb") as f:
            pickle.dump(patch.patch.cpu(), f)
        
    except Exception as e:
        log.write(f'error occured: {e}', _print=True)
        log.save()
        with open(args.resultdir + "/patch(got_errored).pkl", "wb") as f:
            pickle.dump(patch.patch.cpu(), f)