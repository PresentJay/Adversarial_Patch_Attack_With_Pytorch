import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import traceback


def patch_validator(patch, classifier, dataloader, args, log, keep_log=False):
    EOT_FACTORS = {
        "scale" : (args.min_scale, args.max_scale),
        "rotation": (args.min_rotation, args.max_rotation)
    }
    
    try:
        log.write(f'start validate patches with {args.iter_val} data iterations of validation set', _print=True)
        patch.measure_attackCapability(classifier=classifier, val_loader=dataloader, eot_dict=EOT_FACTORS, iteration=args.iter_val)
        log.write(f'Randomly patched model {classifier.getName()} has {patch.attackedAccuracy:.2f}% of accuracy to original, ', end='', _print=True)
        log.write(f'and {patch.attackCapability:.2f}% of accuracy to target {args.target_class}', _print=True)
    except Exception as e:
        log.write(f'error occured during validating: {traceback.format_exc()}', _print=True)
        log.save()
    finally:
        if keep_log:
            log.save()
        

def per_scale_validator(patch, classifier, dataloader, args, log, interval, keep_log=False):
    try:
        aac_list = []
        ac_list = []
        for item in interval:
            eot_factor = {
                "scale" : (-item, item),
                "rotation": (args.min_rotation, args.max_rotation)
            }
            patch.measure_attackCapability(val_loader=dataloader, classifier=classifier, eot_dict=eot_factor, iteration=args.iter_val)
            aac_list.append(patch.attackedAccuracy)
            ac_list.append(patch.attackCapability)
            log.write(f"At {item * 100}% scaled patched image has {patch.attackedAccuracy:.2f}% as accuracy to original, ", end='', _print=True)
            log.write(f'and {patch.attackCapability:.2f}% as attackCapability to class {args.target_class}', _print=True)        
    except Exception as e:
        log.write(f'error occured during validating: {traceback.format_exc()}', _print=True)
        log.save()
    finally:
        if keep_log:
            log.save()
        return aac_list, ac_list


def per_rot_validator(patch, classifier, dataloader, args, log, interval, keep_log=False):
    try:
        aac_list = []
        ac_list = []
        for item in interval:
            eot_factor = {
                "scale" : (args.min_scale, args.max_scale),
                "rotation": (item, item)
            }
            patch.measure_attackCapability(val_loader=dataloader, classifier=classifier, eot_dict=eot_factor, iteration=args.iter_val)
            aac_list.append(patch.attackedAccuracy)
            ac_list.append(patch.attackCapability)
            log.write(f"At {item * 100} degrees rotated patched image has {patch.attackedAccuracy:.2f}% as accuracy to original, ", end='', _print=True)
            log.write(f'and {patch.attackCapability:.2f}% as attackCapability to class {args.target_class}', _print=True)        
    except Exception as e:
        log.write(f'error occured during validating: {traceback.format_exc()}', _print=True)
        log.save()
    finally:
        if keep_log:
            log.save()
        return aac_list, ac_list