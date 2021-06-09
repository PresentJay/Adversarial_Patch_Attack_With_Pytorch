"""
Reference:
    Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

# TODO: apply Google style Python Docstring

import torch
from src import configs, datasets, models, patches
from utils import imgUtil


if __name__ == '__main__':
    # load the Network Settings
    args = configs.Initialization()
    MODELLIST = ["resnet50", "vgg19"]
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
    model_list = models.ModelContainer()
    for model in MODELLIST:
        NetClassifier = models.Model(name=model, dataset=DataSet, device=args.device, explain=args.showProgress, isTorchvision=True)
        model_list.add_model(NetClassifier)
        
    # TODO: concerns to logging!!!!!!!!!!
    # test the original models
    # model_list.test_models(original=True)
    
    for epoch in range(args.epochs):
        patch = patches.AdversarialPatch(dataset=DataSet, target=args.target, device=args.device, _type=args.patch_type)