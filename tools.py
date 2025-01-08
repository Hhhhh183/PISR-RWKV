import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def mkdir(p, is_file=False):
    if is_file:
        p, _ =  os.path.split(p)
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p)) 

def load_bin(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data 
