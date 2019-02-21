import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutil
from matplotlib import pyplot as plt
from util import *


class Dataset_mine(Dataset):
    def __init__(self, root, transform = None):
        super(Dataset_mine, self).__init__()
        self.transform = transform
        train_dir = os.path.join(root,'train')
        test_dir = os.path.join(root, 'test')
        train_list = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
        test_list =  [os.path.join(test_dir, img) for img in os.listdir(test_dir)]
        self.img_list = train_list + test_list
        self.img_list.sort()
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = plt.imread(self.img_list[idx])
        if self.transform:
            img = self.transform(img)
        return img

def unit_test():
    data = Dataset_mine('../face_data', 
                        transform= transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
                        ]))
    loader = DataLoader(data, batch_size= 64)
    imgs = next(iter(loader))
    
    print('data set size:', len(data))
    for id, imgs in enumerate(loader):
        if id == 2:
            break
        print(imgs.size()) 
        print(imgs[0])
        
    plt.figure(figsize= (8, 8))
    plt.imshow(np.transpose(vutil.make_grid(imgs[:64], padding=2,
                normalize= True), (1, 2, 0)))
    plt.show()

    

if __name__ == '__main__':
    unit_test()