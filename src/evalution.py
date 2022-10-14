import numpy as np
import pandas as pd
import os
from os.path import join as opj
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import f1_score, accuracy_score

from tqdm import tqdm
import neptune

from .models import *
from .utils import *

class Tester:
    def __init__(self, test_loader, args):
        
        self.args        = args
        self.criterion   = self.select_loss().to(args.device)
        self.model       = CRFormer(args).to(args.device)
        self.test_loader = test_loader
        self._load_checkpoint()
        
    def select_loss(self):
        
        if self.args.loss=='l1':
            criterion = nn.L1Loss()
        elif self.args.loss=='sl1':
            criterion = nn.SmoothL1Loss()
        elif self.args.loss=='l2':
            criterion = nn.MSELoss()
        return criterion

    def _load_checkpoint(self):
        model_path = opj(self.args.model_path, f'{self.args.objective}_{self.args.sequence_length}_{self.args.model_name}')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print('---load previous weigths and optimizer---')

    def _write_image(self):
        pred_path  = opj(self.args.data_path, 'prediction', self.args.ex_name, self.args.objective, str(self.args.sequence_length))
        image_path = opj(pred_path, 'images')
        MakeDir(image_path)
        
        pred_df    = fastRead_csv(opj(pred_path, 'prediction.csv'))
        label_data = self.test_loader.dataset.days_data.sum(-1)
        keys       = list(self.test_loader.dataset.days_info['index'])
        print('--- Writing Images ---')
        for i in tqdm(range(len(keys))):
            tmp_key   = keys[i]
            tmp_label = label_data[i]
            tmp_pred  = np.array(pred_df[pred_df['index']==tmp_key].iloc[0,:61], dtype=int)
            file_name = tmp_key + '.png'
            plt.plot(tmp_label)
            plt.plot(np.round(tmp_pred))
            plt.title(tmp_key)
            plt.legend(['label', 'pred'])
            plt.savefig(opj(image_path, file_name))
            plt.close()
        
    def test(self):
        
        total_loss = 0 
        total_mae  = 0
        total_rmse = 0
        
        file_list  = []
        label_list = []
        
        input_sequence   = []
        preds_sequence   = []
        pattern_sequence = []

        self.model.eval()
        with torch.no_grad():
            for i, (days_data, cluster_label, pattern_label, features, info) in enumerate(tqdm(self.test_loader)):
                
                input_data_days = days_data[:,:self.args.sequence_length].float().to(self.args.device)
                features        = features.long().to(self.args.device)
                label           = days_data[:,self.args.sequence_length:].float().to(self.args.device)  
                label           = label.sum(dim=-1)
                
                if not self.args.use_feat:
                    features = None
                
                outputs = self.model(input_data_days, features)
                loss    = self.criterion(outputs, label)
                rmse    = torch.sqrt(F.mse_loss(outputs, label))
                mae     = F.l1_loss(outputs, label)
 
                total_loss += loss.item()
                total_mae  += mae.item()
                total_rmse += rmse.item()
                
                outputs    = torch.cat([input_data_days.sum(dim=-1), outputs], dim=-1)
                pred       = outputs.detach().cpu().numpy()
                
                for p in range(len(pred)):
                    label_list.append(pred[p])
                    file_list.append(info[p])
                
                input_sequence.append(days_data.float().sum(dim=-1))
                preds_sequence.append(outputs.detach().cpu())
                pattern_sequence.append(pattern_label)
                
            mae0, rmse0, mae1, rmse1, mae2, rmse2, acc, f1 = self._cal_patten_metrics(input_sequence, preds_sequence, pattern_sequence)
                    
            print("test rmse: {:.4f} | test mae: {:.4f} | test acc: {:.4f} | test f1 {:.4f}".format(total_rmse/(i+1), total_mae/(i+1), acc, f1))
            print("test mae0: {:.4f} | test mae1: {:.4f} | test mae2: {:.4f} | test rmse0: {:.4f} | test rmse1: {:.4f} | test rmse2: {:.4f}".format(mae0, mae1, mae2, rmse0, rmse1, rmse2))
            
            
            if self.args.write_result:
                prediction  = np.stack(label_list)
                index       = np.stack(file_list)
                df          = pd.DataFrame(prediction)
                df['index'] = index
                data_path   = opj(self.args.data_path, 'prediction', self.args.ex_name, self.args.objective, str(self.args.sequence_length))
                
                MakeDir(data_path)
                df.to_csv(opj(data_path, 'prediction.csv'), index=False)
                print('---- Prediction saved ----')
                
            if self.args.logging:
                neptune.log_metric('test loss_final', total_loss/(i+1))
                neptune.log_metric('test mae_final', total_mae/(i+1)) 
                neptune.log_metric('test rmse_final', total_rmse/(i+1))  
                
                neptune.log_metric('test acc', acc)
                neptune.log_metric('test f1', f1) 
                neptune.log_metric('test mae0', mae0)  
                neptune.log_metric('test mae1', mae1)
                neptune.log_metric('test mae2', mae2) 
                neptune.log_metric('test rmse0', rmse0)  
                neptune.log_metric('test rmse1', rmse1)
                neptune.log_metric('test rmse2', rmse2) 
                
                
    def _cal_patten_metrics(self, input_sequence, preds_sequence, pattern_sequence):
        
        pattern_metrics  = {0:[], 1:[], 2:[]}
        input_sequence   = torch.vstack(input_sequence)
        preds_sequence   = torch.vstack(preds_sequence)
        pattern_sequence = torch.hstack(pattern_sequence)
        
        for pattern_i in range(3):
            idx  = torch.where(pattern_sequence==pattern_i)
            mae  = F.l1_loss(preds_sequence[idx][:,self.args.sequence_length:], input_sequence[idx][:,self.args.sequence_length:])
            rmse = torch.sqrt(F.mse_loss(preds_sequence[idx][:,self.args.sequence_length:], input_sequence[idx][:,self.args.sequence_length:]))
            pattern_metrics[pattern_i] = [mae.item(), rmse.item()]
            
        acc, f1 = self._get_pattern_result(preds_sequence.numpy(), pattern_sequence.numpy())
        
        
        return pattern_metrics[0][0], pattern_metrics[0][1], pattern_metrics[1][0], pattern_metrics[1][1], pattern_metrics[2][0], pattern_metrics[2][1], acc, f1
    
    def _get_pattern_result(self, preds_sequence, pattern_sequence):

        clusters          = np.zeros(len(preds_sequence))
        idx_out           = np.where(preds_sequence.max(axis=1)<=8)
        clusters[idx_out] = 10
        
        idx   = np.where(preds_sequence.max(axis=1)>8)
        seq   = preds_sequence[idx]
        seq   = self._scaling(seq)[:,:,:]
        model = TimeSeriesKMeans(n_jobs=-1)
        model = model.from_pickle(opj(self.args.model_path, self.args.objective+'_cluster.pkl'))
        preds = model.predict(seq)
        
        clusters[idx] = preds
        clusters      = np.vectorize(self.args.c2p.__getitem__)(clusters)
        acc           = accuracy_score(pattern_sequence, clusters)
        f1            = f1_score(pattern_sequence, clusters, average='macro')
        
        return acc, f1

    def _scaling(self, x):
        
        # scaling for cluster model
        size  = x.shape
        x = x.reshape(-1,size[1])

        # sequence min / max
        x_max = np.max(x, axis = 1).reshape(-1,1)
        x_min = np.min(x, axis = 1).reshape(-1,1)
        x_max = np.where(x_max==0, 1, x_max)
        
        scaled_input = ((x-x_min)/(x_max-x_min))
        scaled_input = scaled_input.reshape(-1,size[1],1)

        return scaled_input
