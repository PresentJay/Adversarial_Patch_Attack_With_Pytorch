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
        
    # set device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        torch.manual_seed(args.seed)
    
    # load the dataset
    DataSet = datasets.DataSet(
        source=args.data_dir,
        name="ImageNet",
        shape=args.imageshape,
        trainfull=args.trainfull,
        trainsize=args.trainsize,
        testfull=args.testfull,
        testsize=args.testsize,
        hideProgress=args.hideProgress)
    DataSet.SetDataLoader(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # set the models
    # TODO: concerns to logging!!!!!!!!!!
    model_list = models.ModelContainer()
    for model in MODELLIST:
        NetClassifier = models.Model(name=model, device=args.device, hideProgress=args.hideProgress, isTorchvision=True)
        model_list.add_model(NetClassifier)
        
        # test the original model_list
        # if you load Test Data, each trials returns same test data (but order is different)
        # NetClassifier.test(dataloader=DataSet.GetTestData(), original=True)
        
    for epoch in range(args.epochs):
        patch = patches.AdversarialPatch(dataset=DataSet, target=args.target, device=args.device, _type=args.patch_type, hideProgress=args.hideProgress, random_init=args.random_init)
        
        for model in model_list.get_models():
            patch.train(
                model=model, target=args.target, dataloader=DataSet.GetTrainData(),
                lr=args.lr, prob_threshold=args.probability_threshold, max_iteration=args.max_iteration)
            patch.show()
    