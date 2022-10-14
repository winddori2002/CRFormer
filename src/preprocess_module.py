import pandas as pd
import numpy as np
import time
import pickle
import json
import os
import re
from datetime import datetime
from tqdm import tqdm

from .utils import *

def split_data(input_pairs):
    
    data, cols, keys = input_pairs
    chunk_size = len(keys) # option
    cnt = 0
    dic = {}
    
    for index in range(len(keys)):
        temp_data = data[np.where(data[:,cols.index('key')]==keys[index])]
        temp_data = temp_data[:,:-1].astype(int)
        dic[keys[index]] = temp_data
        
    return dic
            
def check_zero_sequence(data):
    return not bool(np.sum(data))

def make_sequence(input_pairs):
    
    data, cols, keys = input_pairs

    days_dic = {}
    bins = list(range(1, 1832, 30))
    for k in keys:
        seq   = np.zeros(len(bins)-1)
        digit = np.digitize(data[k][:,cols.index('days')], bins, right=False) - 1
        seq_idx, seq_cnt = np.unique(digit, return_counts=True)
        seq_cnt          = seq_cnt.astype(int)
        seq[seq_idx]     = seq_cnt
        days_dic[k]      = seq
        
    dist_dic = {}
    bins = list(range(1, 60002, 1000))
    for k in keys:
        seq      = np.zeros(len(bins)-1)
        tmp_data = data[k]
        tmp_data = tmp_data[np.where(tmp_data[:,cols.index('days')] <= 180)]
        digit    = np.digitize(tmp_data[:,cols.index('mile')], bins, right=False) - 1
        seq_idx, seq_cnt = np.unique(digit, return_counts=True)
        seq_cnt          = seq_cnt.astype(int)
        seq[seq_idx]     = seq_cnt
        dist_dic[k]      = seq
    
    return days_dic, dist_dic
    
def key_join(data, index):
    key_col = np.array([','.join(list(i.astype('str'))) for i in data[:, index]])
    return key_col.reshape(-1,1)

def preprocess(data, cols):
    data = data[np.where((data[:,cols.index('days')].astype(int)<=1825) & (data[:,cols.index('days')].astype(int)>0))]
    data = data[np.where((data[:,cols.index('mile')].astype(int)<=60000) & (data[:,cols.index('mile')].astype(int)>0))]
    data = data[np.where(data[:,cols.index('sys')]!='EV')]
    data = data[np.where(data[:,cols.index('sale_consuming_date')].astype(int)>-1)]
    return data

def check_Dtype(data, cols):
    temp_data = np.char.replace(data[:,[cols.index('md_date'), cols.index('sale_date'), 
        cols.index('repair_date')]].astype(str), '-','')
    data[:,[cols.index('md_date'), cols.index('sale_date'), 
        cols.index('repair_date')]] = temp_data
    return data

def DF_to_Numpy(data):
    
    cols       = list(data.columns)
    save_cols  = ['days', 'md_date', 'sale_date', 'repair_date','mile','sale_consuming_date']
    key_cols   = ['year','car_code','part', 'sys']
    key_index  = [cols.index(c) for c in key_cols]
    save_index = [cols.index(c) for c in save_cols]

    data    = np.array(data)
    data    = check_Dtype(data, cols)
    data    = preprocess(data, cols)
    key_col = key_join(data, key_index)

    data = np.concatenate([data[:,save_index], key_col], axis=1)
    save_cols.append('key')
    
    return data, save_cols

def Dict_to_DF(data, cols):
    keys    = list(data.keys())
    key_arr = np.array([i.split(',') for i in keys])
    val_arr = np.stack(list(data.values()))

    data     = np.concatenate([val_arr, key_arr], axis=1)    
    new_cols = list(np.arange(0,val_arr.shape[-1]))+ ['year','car_code','part','sys']
    data     = pd.DataFrame(data, columns=new_cols)
    return data