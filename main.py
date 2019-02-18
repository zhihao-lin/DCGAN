import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_manager import Manaeger
import sys
sys.path.append('models')
from dataset import Dataset_mine

parser = argparse.ArgumentParser()
parser.add_argument('mode', help='Train/Generate', choices=['train', 'generate'])
parser.add_argument('model_G', help= 'Generative Model')
parser.add_argument('model_D', help= 'Discriminative Model')
parser.add_argument('-id', help= 'Experiment ID, used for saving files', default= 0)
parser.add_argument('-lr', help= 'Learning rate',type=float, default= 1e-3)
parser.add_argument('-batch_size', type= int, default= 64)
parser.add_argument('-epoch_num', type = int, default= 100)
parser.add_argument('-load', help='Weights to be load', default= None)
parser.add_argument('-check_batch_num', help= 'How many batches to show result once', type= int, default=200)
parser.add_argument('-gen_dir', help= 'Directory which stores generated images', default= 'generations/')

args = parser.parse_args()

# Prepare datasets, data loader
dataset_trian = Dataset_mine('../face_data', train= True)
dataset_valid = Dataset_mine('../face_data', train= False)
trian_loader = DataLoader(dataset_trian, batch_size= args.batch_size, shuffle= True)
valid_loader = DataLoader(dataset_valid, batch_size= args.batch_size, shuffle= False)

def get_model(name_G, name_D):
    file_G,  file_D  = __import__(name_G), __import__(name_D)
    model_G, model_D = file_G.Generator(), file_D.Discriminator()
    return model_G, model_D

def main():
    print('main function is running ...')
    model_G, model_D = get_model(args.model_G, args.model_D)
    manager = Manaeger(model_G, model_D, args)
    manager.load_data(trian_loader, valid_loader)
    if args.mode == 'train':
        manager.train()
    elif args.mode == 'generate':
        manager.generate()

if __name__ == '__main__':
    main()
