from os import listdir
import os
import pickle

def get_obj_dirs():
    current_dir = os.path.abspath(os.path.curdir)
    obj_dir=os.path.join(current_dir,"results")
    dirs = []
    for experiment in listdir(obj_dir):
        candidate_dir = os.path.join(obj_dir, experiment)
        if os.path.isdir(candidate_dir):
            for timestamp in listdir(candidate_dir):
                valid_dir = os.path.join(candidate_dir, timestamp)
                for filedir in listdir(valid_dir):
                    if filedir.split('.')[1] == 'pkl':
                        dirs.append(valid_dir)
                        break
    return dirs

    
def update_from_log():
    pass
