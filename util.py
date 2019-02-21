import numpy as np
import torch
import cv2 as cv
import torch.nn as nn
from matplotlib import pyplot as plt

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

def save_images(imgs, shape, name= 'image.png'):
    width, height = shape
    num = imgs.size(0)
    if num != width * height:
        print('** Shape not match ! **')
        return 
    img_save =  np.zeros((width * 64, height * 64, 3), dtype= np.float)
    imgs = imgs.permute(0, 2, 3, 1)
    imgs = imgs.numpy()
    # The RGB channel is reversed after torchvision.transforms.ToTensor(),
    # so it need to be reversed back
    imgs_temp = imgs.copy()
    imgs[:, :, :, 0] = imgs_temp[:, :, :, 2]
    imgs[:, :, :, 2] = imgs_temp[:, :, :, 0]

    for w in range(width):
        for h in range(height):
            img_save[64 * w : 64 * (w + 1), 64 * h : 64 * (h + 1)] = imgs[w * height + h]

    plt.imsave(name, img_save)

def initialize_weights(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test():
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = cv.imread('test.png')
    img = transform(img)

    imgs = torch.empty(16, 3, 64, 64)
    for i in range(16):
        imgs[i] = img
    imgs = (imgs /2) + 0.5
    
    save_images(imgs, (4, 4), 'test_out.png')

def test2():
    from matplotlib import pyplot as plt
    img = plt.imread('test.png')
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    test()
    