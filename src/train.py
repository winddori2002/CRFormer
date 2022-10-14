import os
from os.path import join as opj
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import numpy as np
import json
import yaml
import torch
import neptune
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import f1_score, accuracy_score
from .utils import *
from .models import *

class Trainer:

    def __init__(self, data, args):

        self.args      = args
        self.model     = CRFormer(args).to(args.device)
        self.criterion = self.select_loss().to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate)
        
        self.train_loader = data['train']
        self.val_loader   = data['val']
        self.test_loader  = data['test']

        # logging
        if args.logging:
            print('---logging start---')
            neptune_load(get_params(args))
            
        # checkpoint
        if args.checkpoint:
            self._load_checkpoint()
    
    def _load_checkpoint(self):
        model_path = opj(self.args.model_path, f'{self.args.objective}_{self.args.sequence_length}_{self.args.model_name}')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 
        print('---load previous weigths and optimizer---')

    def _save_checkpoint(self, best_loss):
        checkpoint = {'loss': best_loss,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        model_path = opj(self.args.model_path, f'{self.args.objective}_{self.args.sequence_length}_{self.args.model_name}')
        MakeDir(self.args.model_path)
        torch.save(checkpoint, model_path)   

    def select_loss(self):
        
        if self.args.loss=='l1':
            criterion = nn.L1Loss()
        elif self.args.loss=='sl1':
            criterion = nn.SmoothL1Loss()
        elif self.args.loss=='l2':
            criterion = nn.MSELoss()

        return criterion
        
    def train(self):
        
        best_loss = 1000
        for epoch in range(self.args.epoch):
            
            self.model.train()
            train_loss, train_mae, train_rmse  = self._run_epoch(self.train_loader)

            self.model.eval()
            with torch.no_grad():
                val_loss,  val_mae,  val_rmse  = self._run_epoch(self.val_loader, valid=True)
                test_loss, test_mae, test_rmse = self._run_epoch(self.test_loader, valid=True)

            if val_loss < best_loss:
                best_loss = val_loss
                self._save_checkpoint(best_loss)                  
        
            print("epoch: {:03d} | trn loss: {:.4f} | val loss: {:.4f} | test loss: {:.4f} | trn mae: {:.4f} | val mae: {:.4f} | test mae: {:.4f} | trn rmse: {:.4f} | val rmse: {:.4f} | test rmse: {:.4f}".format(epoch, train_loss, val_loss, test_loss, train_mae, val_mae, test_mae, train_rmse, val_rmse, test_rmse))
            print()

            if self.args.logging:
                neptune.log_metric('train loss', train_loss)
                neptune.log_metric('train mae', train_mae)
                neptune.log_metric('train rmse', train_rmse)
                neptune.log_metric('val loss', val_loss)
                neptune.log_metric('val mae', val_mae) 
                neptune.log_metric('val rmse', val_rmse)    
                neptune.log_metric('test loss', test_loss)
                neptune.log_metric('test mae', test_mae) 
                neptune.log_metric('test rmse', test_rmse)         
                
    def _run_epoch(self, data_loader, valid=False):
        
        total_loss = 0 
        total_mae  = 0
        total_rmse = 0
        
        for i, (days_data, cluster_label, pattern_label, features, info) in enumerate(tqdm(data_loader)):
            
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
            
            if not valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_mae  += mae.item()
            total_rmse += rmse.item()
            
        return total_loss/(i+1), total_mae/(i+1), total_rmse/(i+1)

