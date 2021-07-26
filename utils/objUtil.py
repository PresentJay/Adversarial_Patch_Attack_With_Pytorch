from os import listdir, path
import pickle

def get_obj_dirs():
    current_dir = path.abspath(path.curdir)
    obj_dir=path.join(current_dir,"results")
    dirs = []
    for experiment in listdir(obj_dir):
        candidate_dir = path.join(obj_dir, experiment)
        if path.isdir(candidate_dir):
            for timestamp in listdir(candidate_dir):
                valid_dir = path.join(candidate_dir, timestamp)
                for filedir in listdir(valid_dir):
                    if filedir.split('.')[1] == 'pkl':
                        dirs.append(valid_dir)
                        break
    return dirs

    
def update_from_log():
    pass


def get_errored_objs():
    errored_list = []
    targets = get_obj_dirs()
    for target in targets:
        for file in listdir(target):
            if file == "patch(got_errored).pkl":
                errored_list.append(path.join(target, file))
    return errored_list

print(get_errored_objs())