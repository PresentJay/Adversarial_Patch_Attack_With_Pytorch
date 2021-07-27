# TODO: apply Google style Python Docstring
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import traceback
from src import patches
from utils import objUtil

def patch_genetator(classifier, dataset, args, log, keep_log=True):
    EOT_FACTORS = {
        "scale" : (args.min_scale, args.max_scale),
        "rotation": (args.min_rotation, args.max_rotation)
    }
    
    try:    
        # train adversarial patch
        patch = patches.AdversarialPatch(dataset=dataset, target=args.target_class, device=args.device, random_init=args.random_init)
        log.write(f'start train patches with {args.iter_train} data iterations of trainset', _print=True)
        patch.train(classifier=classifier, train_loader=dataset.GetTrainData(), iteration=args.iter_train, eot_dict=EOT_FACTORS, savedir=args.resultdir)
    except Exception as e:
        log.write(f'error occured during training patch: {traceback.format_exc()}', _print=True)
        if not patch.fullyTrained:
            log.write(f"a patch isn't fully trained. try again later.", _print=True)
        log.save()
    finally:
        objUtil.save_obj(obj=patch, save_dir=args.resultdir, name="patch")
        if not keep_log:
            log.save()    
    return patch