import argparse
import os
from os.path import join as opj
import pandas as pd
from datetime import datetime
from src.preprocess_module import *

data_path = ''
file_name = ''

col_dict = {'':''}

def PrepareData():

    data = fastRead_csv(opj(data_path, file_name))
    data.columns = data.columns.map(col_dict) 
    print(f'Origin Data Shape: {data.shape}')
    data, cols = DF_to_Numpy(data)
    print(f'Processed Data Shape: {data.shape}')

    keys = list(np.unique(data[:,cols.index('key')]))
    data = split_data((data, cols, keys))
    
    return data, cols, keys

def make_days_dist_sequence(data, cols, k):
    seq   = np.zeros((61,20))
    days  = data[k][:,cols.index('days')]//30
    miles = data[k][:,cols.index('mile')]//3001
    days_unique = list(np.unique(days))
    
    for i in days_unique:
        seq_idx, seq_cnt = np.unique(miles[np.where(days==i)], return_counts=True)
        seq[i,seq_idx]   = seq_cnt
        
    seq = np.array([str(list(i)) for i in seq]).reshape(1,-1)
    seq = np.concatenate([seq, np.array(k.split(',')).reshape(-1,4)], axis=1)
    
    return seq.reshape(-1)


def MakeTrainData():
    
    data, cols, keys = PrepareData()
    columns = list(range(0,61)) + ['year', 'car_code', 'part', 'sys']
    df      = pd.DataFrame(data=[make_days_dist_sequence(data, cols, k) for k in keys], columns = columns)
    print(f'Sequence Shape: {df.shape}')
    Write_csv(df, opj(data_path, 'days_dist_df_v3.csv'))

    
if __name__ == '__main__':
    MakeTrainData()
