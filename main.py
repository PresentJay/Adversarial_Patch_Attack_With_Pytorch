"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

# TODO: apply Google style Python Docstring

from src import configs, datasets, models, patches
from utils import imgUtil, logUtil


if __name__ == '__main__':
    # load the Network Settings
    args = configs.Initialization()
    
    log = logUtil.log(args.name, args.log_type)
    log.write(f'Random Seed: {args.seed}', _print=True)
    log.write(f'rotation: {args.min_angle}-{args.max_angle} degree', _print=True)
    log.write(f'scale: {args.min_scale*100}%-{args.max_scale*100}%', _print=True)
    
    # set the models
    classifier = models.Model(name=args.model, device=args.device, isTorchvision=args.torchvision)
    
    # load the dataset
    DataSet = datasets.DataSet(train=args.train_dir, val=args.val_dir, name="ImageNet", shape=args.imageshape)
    DataSet.SetDataLoader(batch_size=args.batch_size, num_workers=args.num_workers)
    
    patch = patches.AdversarialPatch(dataset=DataSet, target=args.target, device=args.device, random_init=args.random_init)     
    patch.train(model=args.model, dataloader=DataSet.GetTrainData(), iter=args.iter)
    patch.show()
    