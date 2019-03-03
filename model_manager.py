import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util import *

class Manager():
    def __init__(self, model_G, model_D, args):
        
        if  args.load:
            load_name = os.path.join('../weights/', args.load)
            model_G.load_state_dict(torch.load(load_name + '_G.pkl'))
            model_D.load_state_dict(torch.load(load_name + '_D.pkl'))
        else:
            model_G.apply(initialize_weights)
            model_D.apply(initialize_weights)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_G = model_G.to(self.device)
        self.latent_dim = model_G.get_latent_dim()
        self.model_D = model_D.to(self.device)
        self.id = args.id
        self.lr = args.lr
        self.optimizer_G = optim.Adam(self.model_G.parameters(), lr= self.lr, betas= (args.beta, 0.999))
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr= self.lr, betas= (args.beta, 0.999))
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.bce_loss = nn.BCELoss()
        
        self.save_name_G = os.path.join('../weights/', self.id + '_G.pkl') 
        self.save_name_D = os.path.join('../weights/', self.id + '_D.pkl') 
        self.log_file = open('logs/' + self.id + '.txt', 'w')
        self.check_batch_num = args.check_batch_num
        self.gen_dir = os.path.join('generations', args.id)
        if not os.path.isdir(self.gen_dir):
            os.mkdir(self.gen_dir)
        self.info = args.info
    
    def load_data(self, data_loader):
        self.data_loader = data_loader
        
    def record(self, message):
        self.log_file.write(message)
        print(message, end='')

    def get_info(self):
        info = get_string('\nID:', self.id, '\n')
        info = get_string(info, 'infomation:', self.info, '\n')
        info = get_string(info, 'Generator:', self.model_G.name(), '\n')
        info = get_string(info, 'Discriminator:', self.model_D.name(), '\n')
        info = get_string(info, 'Learning rate:', self.lr, '\n')
        info = get_string(info, 'Epoch number:', self.epoch_num, '\n')
        info = get_string(info, 'Batch size:', self.batch_size, '\n')
        info = get_string(info, '=======================\n\n')
        return info

    def train(self):
        info = self.get_info()
        self.record(info)
        self.record(self.model_G.__str__() + '\n')
        self.record(self.model_D.__str__() + '\n')

        fix_noise = torch.randn(self.batch_size, self.latent_dim, 1, 1).to(self.device)
        label_real = 1
        lable_fake = 0
        labels = torch.empty(self.batch_size, 1).to(self.device)

        for epoch in range(self.epoch_num):
            self.model_G.train()
            self.model_D.train()
            
            
            for batch_id, imgs in enumerate(self.data_loader):
                ###################
                # Update D net work
                ###################
                self.model_D.zero_grad()
                real_imgs = imgs.to(self.device)

                # train with real images
                labels.fill_(label_real)
                output = self.model_D(real_imgs)
                loss_real = self.bce_loss(output, labels)
                loss_real.backward()

                # train with fake images
                labels.fill_(lable_fake)
                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1).to(self.device)
                fake_imgs = self.model_G(noise)
                output = self.model_D(fake_imgs.detach())
                loss_fake = self.bce_loss(output, labels)
                loss_fake.backward()
                
                self.optimizer_D.step()

                ###################
                # Update G net work
                ###################
                self.model_G.zero_grad()
                labels.fill_(label_real)
                output = self.model_D(fake_imgs)
                loss_gen = self.bce_loss(output, labels)
                loss_gen.backward()
                self.optimizer_G.step()

                # Record some information
                if (batch_id + 1) % self.check_batch_num == 0:
                    info = get_string('Epoch ', epoch, 'Step', batch_id + 1, 
                                      '| G loss :', loss_gen.item()/self.batch_size,
                                      '| D loss (real) : ', loss_real.item()/self.batch_size,
                                      '| D loss (fake) : ', loss_fake.item()/self.batch_size)
                    self.record(info + '\n')

            check_imgs = self.model_G(fix_noise)
            check_imgs = (check_imgs / 2) + 0.5
            check_imgs = check_imgs.cpu().detach()
            save_images(check_imgs, (8, 8), os.path.join(self.gen_dir, 'Epoch_' + str(epoch) + '.png'))

            torch.save(self.model_G.state_dict(), self.save_name_G)
            torch.save(self.model_D.state_dict(), self.save_name_D)
                    

    def generate(self):
        print('Generating ... ')
        self.model_G.eval()
        noise = torch.randn(self.batch_size, self.latent_dim, 1, 1).to(self.device)
        imgs_gen = self.model_G(noise)
