"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

# TODO: apply Google style Python Docstring

import torch
from src import configs, datasets, models, patches


if __name__ == '__main__':
    # load the Network Settings
    args = configs.Initialization()
    MODELLIST = ["resnet50"]
    DATASET = "imagenet"
        
    # set device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.manualSeed)
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        torch.manual_seed(args.manualSeed)
    
    # load the dataset
    # TODO: apply statusbar
    DataSet = datasets.DataSet(
        source=args.data_dir,
        name="ImageNet",
        shape=args.imageshape,
        trainfull=args.trainfull,
        trainsize=args.trainsize,
        testfull=args.testfull,
        testsize=args.testsize,
        explain=args.showProgress)
    DataSet.SetDataLoader(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # set the models
    ModelContainer = models.ModelContainer()
    for model in MODELLIST:
        NetClassifier = models.Model(name=model, dataset=DataSet, device=args.device, explain=args.showProgress, isTorchvision=True)
        ModelContainer.add_model(NetClassifier)