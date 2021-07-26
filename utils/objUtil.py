import pickle
import shutil
from os import listdir, path

def get_obj_dirs(detail=False, filter = 'pkl'):
    current_dir = path.abspath(path.curdir)
    obj_dir=path.join(current_dir,"results")
    dirs = []
    if detail:
        objs = []
    for experiment in listdir(obj_dir):
        candidate_dir = path.join(obj_dir, experiment)
        if path.isdir(candidate_dir):
            for timestamp in listdir(candidate_dir):
                valid_dir = path.join(candidate_dir, timestamp)
                for filedir in listdir(valid_dir):
                    if filedir.split('.')[1] == filter:
                        dirs.append(valid_dir)
                        if detail:
                            objs.append(path.join(valid_dir, filedir))
                        break
    if detail:
        return objs
    return dirs


def get_object(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
        return data


# TODO: 
def update_from_log():
    pass


# temporary method . . . TODO: deletion if it is useless
def get_errored_objs(delete = True):
    errored_list = []
    targets = get_obj_dirs()
    for target in targets:
        for file in listdir(target):
            if file == "patch(got_errored).pkl":
                errored_list.append(path.join(target, file))
                if delete:
                    shutil.rmtree(target)
                    print(f"delete occur in {target}")
    if delete:
        return []
    return errored_list