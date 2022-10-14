import os
from os.path import join as opj
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import re
from ast import literal_eval
from .utils import *

def Info2Idx(train_df_info, features):
    info2idx = {}
    for f in features:
        f_unique    = train_df_info[f].unique()
        info2idx[f] = {k:v+1 for v, k in enumerate(f_unique)}
    return info2idx

def MappingInfo(df, info2idx, features):
    for f in features:
        df[f] = df[f].map(info2idx[f])
    df = df.fillna(0)
    return df

def SetScaleData(raw_sales_df_dir, data_path):
    mysales = fastRead_sales(opj(raw_sales_df_dir))
    cols = [re.findall('\(([^)]+)',col) for col in mysales.columns]
    cols[0], cols[1] = ['year'],['date']
    cols = [y.upper() for x in cols for y in x]
    mysales.columns = cols

    for idx, row in mysales.iterrows():
        mysales.iloc[idx,:] = row.apply(lambda x : '0' if x == "" else x).str.replace(',','')
    mysales.iloc[:,2:] = mysales.iloc[:,2:].apply(lambda x: x.astype(int))

    tmp = mysales.groupby('YEAR').sum()

    year_lst = []
    key_lst = []
    val_lst = []
    for idx, row in tmp.iterrows():
        for i in range(sum(row != 0)):
            year_lst.append(idx)
            key_lst.append(row[(row != 0)].keys()[i])
            val_lst.append(int(row[(row != 0)].values[i]))

    sales_df = pd.DataFrame([x for x in zip(key_lst, year_lst, val_lst)])
    sales_df.columns = ['car_code','year', 'total_sales']

    sales_df['car_code'] = sales_df['car_code'].replace('DMA', 'H1')
    sales_df['car_code'] = sales_df['car_code'].replace('PS', 'H2')
    sales_df['car_code'] = sales_df['car_code'].replace('QF', 'H3')


    Write_csv(sales_df, opj(data_path, 'sales_df.csv'))
    print('------------------Sales Data Save in {}----------------------'.format(opj(data_path, 'sales_df.csv')))
    
    return sales_df

def ScaleData(days_data, days_info, sales_data):

    tmp = [sales_data.loc[np.where((days_info['car_code'].iloc[i] == sales_data['car_code']) & 
          (days_info['year'].iloc[i] == sales_data['year']))[0][0], 'total_sales'] for i in range(days_info.shape[0])]

    days_data = np.stack([i / scales * 10000 for i, scales in zip(days_data, tmp)])

    return days_data

def GetData(df, year):
    
    df['index'] = df[['car_code','year','part','sys']].astype(str).agg(','.join, axis=1)

    idx = []
    for i in year:
        idx += list(df[df["year"]==f"{i}MY"].index)  
    
    df = df.iloc[idx]
    df = df.sort_values(['index'])

    df, df_info = df.iloc[:,:61], df.iloc[:,61:]
        
    return df, df_info

def GetDaysDistData(data):
    for i in range(61):
        data[str(i)] = data[str(i)].apply(lambda x: literal_eval(x))
    data = np.array(data.values.tolist())
    return data

class TrainDataset(Dataset):
    def __init__(self, args):
        
        if args.objective=='days':
            df     = fastRead_csv(opj(args.data_path, 'days_dist_df.csv'))
            tmp_df = fastRead_csv(opj(args.data_path, 'days_df_cluster.csv'))
        else:
            df     = fastRead_csv(opj(args.data_path, 'dist_days_df.csv'))
            tmp_df = fastRead_csv(opj(args.data_path, 'dist_df_cluster.csv'))
        
        self.days_data, self.days_info = GetData(df, args.train_year)
        self.tmp_data, self.tmp_info   = GetData(tmp_df, args.train_year)
        self.days_data                 = GetDaysDistData(self.days_data)
        self.pattern_label             = np.array(self.tmp_info['label'].map(args.c2p))
        self.cluster_label             = np.array(self.tmp_info['label'].astype(int))

        if args.scaling:
            self.sales_data = SetScaleData(args.sales_path, args.data_path)
            self.days_data  = ScaleData(self.days_data, self.days_info, self.sales_data)
            
        self.info2idx       = Info2Idx(self.days_info, features=args.features)
        self.days_info      = MappingInfo(self.days_info, self.info2idx, args.features)
        self.extra_features = np.array(self.days_info[args.features])    
        self.seq_idx_days = np.array(self.days_info['index'])
        
    def __len__(self):
        return len(self.days_data)

    def __getitem__(self, index):
        
        days_data     = self.days_data[index]
        cluster_label = self.cluster_label[index]
        pattern_label = self.pattern_label[index]
        features      = self.extra_features[index]
        info          = self.seq_idx_days[index]

        return days_data, cluster_label, pattern_label, features, info
    
class TestDataset(Dataset):
    def __init__(self, info2idx, args, val=True):
        
        if args.objective=='days':
            df     = fastRead_csv(opj(args.data_path, 'days_dist_df.csv'))
            tmp_df = fastRead_csv(opj(args.data_path, 'days_df_cluster.csv'))
        else:
            df     = fastRead_csv(opj(args.data_path, 'dist_days_df.csv'))
            tmp_df = fastRead_csv(opj(args.data_path, 'dist_df_cluster.csv'))
        
        if val:
            years = args.val_year
        else:
            years = args.test_year

        self.days_data, self.days_info = GetData(df, years)   
        self.tmp_data, self.tmp_info   = GetData(tmp_df, years)
        self.days_data                 = GetDaysDistData(self.days_data)
        self.pattern_label             = np.array(self.tmp_info['label'].map(args.c2p))
        self.cluster_label             = np.array(self.tmp_info['label'].astype(int))
                   
        if args.scaling:
            self.sales_data = fastRead_csv(opj(args.data_path, 'sales_df.csv'))
            self.days_data  = ScaleData(self.days_data, self.days_info, self.sales_data)
        self.days_info      = MappingInfo(self.days_info, info2idx, args.features)
        self.extra_features = np.array(self.days_info[args.features])
        self.seq_idx_days   = np.array(self.days_info['index'])
        
    def __len__(self):
        return len(self.days_data)

    def __getitem__(self, index):
        
        days_data     = self.days_data[index]
        cluster_label = self.cluster_label[index]
        pattern_label = self.pattern_label[index]
        features      = self.extra_features[index]
        info          = self.seq_idx_days[index]

        return days_data, cluster_label, pattern_label, features, info