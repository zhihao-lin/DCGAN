import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 512

    def name(self):
        return 'g_01'

    def get_latent_dim(self):
        return self.latent_dim

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