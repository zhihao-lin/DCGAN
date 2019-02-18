import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        

    def name(self):
        return 'd_01'

    def forward(self, x):
        return x
        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    batch_size = 64
    print('Parameter number: ',parameter_number())
    print('Input size: ', )
    print('Output size:', )
    
if __name__ == '__main__':
    unit_test()