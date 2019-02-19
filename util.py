import numpy as np
import torch
import cv2 as cv

def get_string(*args):
    string = ''
    for s in args:
        string = string + ' ' + str(s)
    return string

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
    img = read_img_to_tensor('test.png')
    print(img.size())
    print(img)
    img = image_tensor_to_numpy(img)
    print(img.shape)
    print(img)
    cv.imshow('test',img)
    cv.waitKey(0)

if __name__ == '__main__':
    test()
    