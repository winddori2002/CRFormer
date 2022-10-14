import os 
import argparse

def get_config():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('action', type=str, default='train', help='Action') # train / test
    
    # dataset
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--sales_path', default='sales.csv', type=str)
    parser.add_argument('--train_year', default=['06','07','08','09','10','11','12','13','14'], nargs='+')
    parser.add_argument('--val_year', default=['15'], nargs='+')
    parser.add_argument('--test_year', default=['16'], nargs='+')
    parser.add_argument('--sequence_length', default=6, type=int)
    parser.add_argument('--objective', default='days', type=str)

    #model - crformer
    parser.add_argument('--scaling', default=False, type=bool) # {True: scaling by sales}
    parser.add_argument('--input_size', default=1, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--num_layers', default=6, type=int)
    parser.add_argument('--d_ffn', default=256, type=int)  
    parser.add_argument('--n_head', default=4, type=int)  
    parser.add_argument('--d_k', default=64, type=int)  
    parser.add_argument('--dropout', default=0.3, type=float)  
    parser.add_argument('--use_feat', default=False, type=bool)
    parser.add_argument('--features', default=['sys','car_code','part'], nargs='+')

    #basic
    parser.add_argument('--write_result', type=bool, default=False, help='Write prediction csv for test')
    parser.add_argument('--write_image', type=bool, default=False, help='Write image for test')
    parser.add_argument('--model_path', type=str, default='./weights', help='Model path')
    parser.add_argument('--model_name', type=str, default='model.pth', help='Model name')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--loss', type=str, default='l2', help='Loss') # {l1:L1 loss, ch:chabonnier loss}
    parser.add_argument('--checkpoint', type=bool, default=False, help='Checkpoint') # If you want to train with pre-trained, or resume set True

    # device 
    parser.add_argument('--device', type=str, default='cuda:0', help='Gpu device')
    parser.add_argument('--env', type=str, default='local', help='Enviornment')
    parser.add_argument('--num_worker', type=int, default=0, help='Num workers')

    # logging setting
    parser.add_argument('--logging', type=bool, default=False, help='Logging')
    parser.add_argument('--tag', type=str, nargs='+', default=None, help="Experiment Tags")
    arguments = parser.parse_args()
    
    return arguments
