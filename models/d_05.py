import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 64
        
        self.model = nn.Sequential(
            # input (b, 3, 64, 64)
            nn.Conv2d(3, ndf, 3, 1, 1, bias= False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace= True),
            nn.MaxPool2d((2, 2), (2, 2)),
            # (b, ndf, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias= False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace= True),
            nn.MaxPool2d((2, 2), (2, 2)),
            # (b, ndf * 2, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias= False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace= True),
            nn.MaxPool2d((2, 2), (2, 2)),
            # (b, ndf * 4, 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias= False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace= True),
            nn.MaxPool2d((2, 2), (2, 2)),
            # (b, ndf * 8, 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias= False),
            nn.Sigmoid()
        )

    def name(self):
        return 'Use Maxpooling for down-sampling'

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
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