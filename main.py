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
    args = configs.init_args()
    MODELLIST = ["resnet50"]
    DATASET = "imagenet"
        
    # set device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    
    # load the dataset
    # TODO: apply statusbar
    DataSet = datasets.DataSet(source=args.data_dir, name="ImageNet", shape=[3, args.image_size, args.image_size], size=args.datasize,, explain=args.showProgress)
    
    # set the model
    ModelContainer = models.ModelContainer(device=args.device)
    for model in MODELLIST:
        NetClassifier = models.Model(name=model, explain=args.showProgress, isTorchvision=True)
        ModelContainer.add_model(NetClassifier.to(args.device))
        
    
    
    
        
        