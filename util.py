import numpy as np
import torch
import cv2 as cv

def read_img_to_tensor(path):
    img = cv.imread(path)
    img = torch.tensor(img, dtype= torch.float)
    img = img.permute(2, 0, 1)
    return img

def image_tensor_to_numpy(tensor):
    tensor = tensor.permute(1, 2, 0).type(torch.uint8)
    img = tensor.cpu().numpy()
    return img

def test():
    pass

if __name__ == '__main__':
    test()
    