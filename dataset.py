import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from util import *
import os

class Dataset_mine(Dataset):
    def __init__(self, root):
        super(Dataset_mine, self).__init__()
        
        
        train_dir = os.path.join(root,'train')
        test_dir = os.path.join(root, 'test')
        train_list = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
        test_list =  [os.path.join(test_dir, img) for img in os.listdir(test_dir)]
        self.img_list = train_list + test_list
        self.img_list.sort()
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = read_img_to_tensor(self.img_list[idx])
        return img

def unit_test():
    data = Dataset_mine('../face_data')
    loader = DataLoader(data, batch_size= 8)
    
    print('data set size:', len(data))
    for id, imgs in enumerate(loader):
        if id == 2:
            break
        print(imgs.size()) 

if __name__ == '__main__':
    unit_test()