import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        channels = [3, 64, 256, 512]

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size= 3, padding= 1)
        self.pool1 = nn.MaxPool2d((4, 4), (4, 4))
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size= 3, padding= 1)
        self.pool2 = nn.MaxPool2d((4, 4), (4, 4))
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size= 3, padding= 1)
        self.pool3 = nn.MaxPool2d((4, 4), (4, 4))
        
        self.fc1 = nn.Linear(channels[-1] * 1 * 1, 8)
        self.fc2 = nn.Linear(8, 1)

    def name(self):
        return 'd_02'

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    imgs = torch.zeros(10, 3, 64, 64)
    model = Discriminator()
    out = model(imgs)

    print('Parameter number: ',parameter_number(model))
    print('Input size: ', imgs.size())
    print('Output size:', out.size())
    
if __name__ == '__main__':
    unit_test()