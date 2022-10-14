import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from os.path import join as opj
import joblib
from tqdm import tqdm
import argparse
from tslearn.clustering import TimeSeriesKMeans
from src.utils import *
import warnings
warnings.filterwarnings(action='ignore')

class TimeSeriesCluster:
    """
    TimeSeries Kmeans Cluster
    This is for obtaining pattern cluster information from claim sequences for pattern-wise evaluation.
    Expected patterns are irregular, decreasing, and increasing failure patterns.
    """
    
    def __init__(self, args):
        
        self.train       = args.train
        self.images      = args.write_image
        self.max_image   = args.max_image
        self.data_path   = args.data_path
        self.image_path  = opj(args.data_path, 'cluster')
        self.weight_path = opj(args.model_path)
        self.n_cluster   = args.n_cluster
        self.days_df     = fastRead_csv(opj(args.data_path, 'days_df.csv'))

        MakeDir(self.image_path)
        MakeDir(self.weight_path)
        self.weight_path = opj(self.weight_path, 'cluster.pkl')
        
    def _scaling(self, input):

        size  = input.shape
        input = input.reshape(-1,size[1])

        # sequence min / max
        x_max = np.max(input, axis=1).reshape(-1,1)
        x_min = np.min(input, axis=1).reshape(-1,1)
        x_max = np.where(x_max==0, 1, x_max)
        
        scaled_input = ((input-x_min)/(x_max-x_min))
        scaled_input = scaled_input.reshape(-1,size[1],1)

        return scaled_input

    def _write_image(self):
        
        days_df       = fastRead_csv(opj(self.data_path, 'days_df_cluster.csv'))
        cluster_list  = list(days_df['label'].unique())
        seq, seq_info = self._split_data(days_df)
        cluster       = np.array(days_df['label'].astype(int))
        [MakeDir(opj(self.image_path, str(i))) for i in cluster_list]
        
        for i in tqdm(cluster_list):
            idx      = np.where(cluster==i)
            tmp_seq  = seq[idx]
            tmp_info = seq_info[idx]
            
            for j in range(len(tmp_seq)):
                fig = plt.figure()
                plt.plot(tmp_seq[j])
                plt.title(tmp_info[j])
                plt.savefig(opj(self.image_path, str(i),tmp_info[j])+'.png')
                plt.close(fig)
                a += 1

    def _split_data(self, input):
        
        input['index'] = input[['car_code','year','part','sys']].astype(str).agg(','.join, axis=1)
        seq            = np.array(input.iloc[:,:61].astype(float))
        seq_info       = np.array(input['index'])
            
        return seq, seq_info
    
    def _infer(self, input):
        
        clusters          = np.zeros(len(input))
        idx_out           = np.where(input.max(axis=1)<=10)
        clusters[idx_out] = self.n_cluster
        
        idx   = np.where(input.max(axis=1)>10)
        seq   = input[idx]
        seq   = self._scaling(seq)[:,:,:]
        model = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=100, n_jobs=-1, random_state=100)
        model = model.from_pickle(self.weight_path)
        preds = model.predict(seq)
        
        clusters[idx]         = preds
        self.days_df['label'] = clusters
        
        Write_csv(self.days_df, opj(self.data_path, 'days_df_cluster.csv'))
        print('---cluster data saved---')
        
    def _fit_cluster(self, input):
        
        idx   = np.where(input.max(axis=1)>10)
        seq   = input[idx]
        seq   = self._scaling(seq)[:,:,:]
        model = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=100, n_jobs=-1, random_state=100).fit(seq)
        
        model.to_pickle(self.weight_path)
        print('--- cluster model saved ---')

    def forward(self, x):
        
        seq, seq_info = self._split_data(x)
        if self.train:
            self._fit_cluster(seq)
        else:
            self._infer(seq)
        if self.images:
            self._write_image()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)  # days / dist
    parser.add_argument('--data_path', type=str, default='')  # dura / qual
    parser.add_argument('--model_path', type=str, default='./weights') 
    parser.add_argument('--n_cluster', type=int, default=10) 
    parser.add_argument('--write_image', type=bool, default=False)
    parser.add_argument('--max_image', type=int, default=200) 
  
    args = parser.parse_args()
    print(vars(args))
 
    days_df = fastRead_csv(opj(args.data_path, 'days_df.csv'))
    tsc     = TimeSeriesCluster(args)
    tsc.forward(days_df)