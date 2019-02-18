import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util import *

class Dataset_mine(Dataset):
    def __init__(self, root, train = True):
        super(Dataset_mine, self).__init__()
        data_dir = None
        if train == True:
            data_dir = os.path.join(root,'train')
        else:
            data_dir = os.path.join(root, 'test')
        
        self.img_list = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
        self.img_list.sort()
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = read_img_to_tensor(self.img_list[idx])
        return img

def unit_test():
    data_train = Dataset_mine('../face_data', train=True)
    data_valid = Dataset_mine('../face_data', train= False)
    loader = DataLoader(data_train, batch_size= 8)
    
    for id, imgs in enumerate(loader):
        if id == 2:
            break
        print(imgs.size()) 

if __name__ == '__main__':
    unit_test()