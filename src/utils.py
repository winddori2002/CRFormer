import numpy as np
import pandas as pd
import os
from pyarrow import csv
import _pickle as cPickle
import random
import torch
import neptune
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

def MakeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def neptune_load(PARAMS):
    """
    logging: write your neptune account/project, api topken
    """
    neptune.init('ID/Project', api_token = 'api_token')
    neptune.create_experiment(name=PARAMS['ex_name'], params=PARAMS)
    if PARAMS['tag'] is not None:
        for tagged in PARAMS['tag']:
            neptune.append_tag(str(tagged))
            
def get_params(args):
    
    params    = {}
    args_ref  = vars(args)
    args_keys = vars(args).keys()

    for key in args_keys:
        if '__' in key:
            continue
        else:
            temp_params = args_ref[key]
            if type(temp_params) == dict:
                params.update(temp_params)
            else:
                params[key] = temp_params
                
    return params

def fastRead_sales(path):
    r_opt = csv.ReadOptions(encoding='cp949', skip_rows = 1, skip_rows_after_names = 1)
    return csv.read_csv(path, read_options=r_opt).to_pandas()

def fastWrite_pkl(data, path):
    with open(path, 'wb') as f:
        cPickle.dump(data, f, protocol=-1)

def fastRead_csv(path):
    r_opt = csv.ReadOptions(encoding='cp949')
    return csv.read_csv(path, read_options=r_opt).to_pandas()

def fastRead_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def Read_csv(path):
    data = pd.read_csv(path)
    return data

def Write_csv(data, path):
    data.to_csv(path, encoding='cp949', index=False)

def seed_init(seed=100):
    
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  