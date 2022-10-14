import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from os.path import join as opj
import pandas as pd
from src.utils import *
from src.dataset import *
from config import *
from src.preprocess_module import *
from src.dataset import *
from src.train import *
from src.evalution import *

def main():
    
    args         = get_config()
    args.ex_name = os.path.basename(os.getcwd())
    # this is for pattern mapping (irregular, decreasing, increasing failure patterns)
    if args.objective=='days':
        args.c2p = {0:2, 1:1, 2:2, 3:1, 4:1, 5:1, 6:1, 7:2, 8:2, 9:2, 10:0}
    else: 
        args.c2p = {0:1, 1:2, 2:2, 3:2, 4:2, 5:1, 6:1, 7:1, 8:2, 9:2, 10:0}
    print(vars(args))
    seed_init()

    if args.action == 'train':
        train_dataset = TrainDataset(args)
        val_dataset   = TestDataset(train_dataset.info2idx, args, val=True)
        test_dataset  = TestDataset(train_dataset.info2idx, args, val=False)
        args.f_sizes  = [len(train_dataset.info2idx[i])+1 for i in args.features]
        
        print('Train years and sizes: ', list(train_dataset.days_info['year'].unique()), len(train_dataset.days_info))
        print('Valid years and sizes: ', list(val_dataset.days_info['year'].unique()),   len(val_dataset.days_info))
        print('Test years and sizes: ',  list(test_dataset.days_info['year'].unique()),  len(test_dataset.days_info))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_worker)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        data_loader  = {'train':train_loader, 'val':val_loader, 'test':test_loader}

        trainer = Trainer(data_loader, args)
        trainer.train()
        
        print('---Test Phase ---')
        tester  = Tester(test_loader, args)
        tester.test()
        
        if args.logging:
            neptune.stop()

    else:
        train_dataset = TrainDataset(args)
        test_dataset  = TestDataset(train_dataset.info2idx, args, val=False)
        test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        args.f_sizes  = [len(train_dataset.info2idx[i])+1 for i in args.features]
        tester        = Tester(test_loader, args)
        tester.test()

        if args.write_image:
            tester._write_image()
    
if __name__ == '__main__':
    main()