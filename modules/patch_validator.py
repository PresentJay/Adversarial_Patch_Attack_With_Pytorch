import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import traceback

def patch_validator(patch, classifier, dataset, args, log):
    EOT_FACTORS = {
        "scale" : (args.min_scale, args.max_scale),
        "rotation": (args.min_rotation, args.max_rotation)
    }
    
    try:
        log.write(f'start validate patches with {args.iter_val} data iterations of validation set', _print=True)
        patch.measure_attackCapability(classifier=classifier, val_loader=dataset.GetValData(), eot_dict=EOT_FACTORS, iteration=args.iter_val)
        log.write(f'Randomly patched model {classifier.getName()} has {patch.attackedAccuracy:.2f}% of accuracy to original, ', end='', _print=True)
        log.write(f'and {patch.attackCapability:.2f}% of accuracy to target {args.target_class}', _print=True)
        log.save()
    except Exception as e:
        log.write(f'error occured during validating: {traceback.format_exc()}', _print=True)
        log.save()
