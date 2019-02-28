import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 100
        ngf = 64

        self.model = nn.Sequential(
            # (b, 100, 1, 1)
            nn.ConvTranspose2d(self.latent_dim, ngf * 8, 4, 1, 0, bias= False),
            nn.ReLU(inplace= True),
            # (b, ngf * 8, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias= False),
            nn.ReLU(inplace= True),
            # (b, ngf * 4, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias= False),
            nn.ReLU(inplace= True),
            # (b, ngf * 2, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias= False),
            nn.ReLU(inplace= True),
            # (b, ngf, 32, 32)
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias= False),
            nn.Tanh()
            # (b, 3, 64, 64)
        )
        

    def name(self):
        return 'g_tutorial without BN'

    def get_latent_dim(self):
        return self.latent_dim

    def forward(self, x):
        x = self.model(x)
        return x
        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    batch_size = 64
    model = Generator()
    vector = torch.zeros(10, model.get_latent_dim(), 1, 1)
    out = model(vector)
    print('Parameter number: ',parameter_number(model))
    print('Input size: ', vector.size())
    print('Output size:', out.size())
    
if __name__ == '__main__':
    unit_test()