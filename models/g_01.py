import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 128
        channels = [512, 256, 128, 64, 3]

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu  = nn.ReLU()
        self.fc1  = nn.Linear(self.latent_dim, 512)
        self.deconv1 = nn.ConvTranspose2d(channels[0], channels[1], kernel_size= 4, stride= 4)
        self.deconv2 = nn.ConvTranspose2d(channels[1], channels[2], kernel_size= 4, stride= 4)
        self.deconv3 = nn.ConvTranspose2d(channels[2], channels[3], kernel_size= 2, stride= 2)
        self.deconv4 = nn.ConvTranspose2d(channels[3], channels[4], kernel_size= 2, stride= 2)

    def name(self):
        return 'g_01'

    def get_latent_dim(self):
        return self.latent_dim

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.view(x.size(0), 512, 1, 1)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        return x
        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    batch_size = 64
    model = Generator()
    vector = torch.zeros(10, model.get_latent_dim())
    out = model(vector)
    print('Parameter number: ',parameter_number(model))
    print('Input size: ', vector.size())
    print('Output size:', out.size())
    
if __name__ == '__main__':
    unit_test()