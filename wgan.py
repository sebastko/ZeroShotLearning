from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.distributions import uniform, normal
import os
import json


class WassersteinGAN:

    def __init__(self, device, x_dim, z_dim, attr_dim, classifier, **kwargs):

        self.device = device

        self.x_dim = x_dim

        #parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
        self.nz = z_dim
        self.attr_dim = attr_dim
        self.classifier = classifier

        self.Z_dist = normal.Normal(0, 1)

        self.batchSize = kwargs.get('batchSize', 64)
        #parser.add_argument('--ngf', type=int, default=64)
        #parser.add_argument('--ndf', type=int, default=64)
        self.niter = kwargs.get('niter', 25) #, help='number of epochs to train for')
        self.lrD = kwargs.get('lrD', 0.00005) #, help='learning rate for Critic, default=0.00005')
        self.lrG = kwargs.get('lrG', 0.00005) #, help='learning rate for Generator, default=0.00005')
        self.beta1 = kwargs.get('beta1', 0.5) #, help='beta1 for adam. default=0.5')
        netG_path = kwargs.get('netG', '') #, help="path to netG (to continue training)")
        netD_path = kwargs.get('netD', '') #, help="path to netD (to continue training)")
        self.clamp_lower = kwargs.get('clamp_lower', -0.01)
        self.clamp_upper = kwargs.get('clamp_upper', 0.01)
        self.Diters = kwargs.get('Diters', 5) #, help='number of D iters per each G iter')
        self.experiment = kwargs.get('experiment', 'samples') #, help='Where to store samples and models')
        self.adam = kwargs.get('adam', False) #, help='Whether to use adam (default is rmsprop)')

        self.lmbda = kwargs.get('lmbda', 10.0)
        self.beta = kwargs.get('beta', 0.01)

        self.model_save_dir = "saved_models"
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        nz = int(self.nz)
        #ngf = int(self.ngf)
        #ndf = int(self.ndf)
        #nc = int(self.nc)
        #n_extra_layers = int(self.n_extra_layers)

        # custom weights initialization called on netG.
        # NOTE: WGAN code doesn't apply it on netD in the MLP case, so we're skipping it for netD too.
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        self.netG = nn.Sequential(
            nn.Linear(self.nz + self.attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, self.x_dim),
            nn.ReLU()
        ).to(self.device)

        self.netG.apply(weights_init)
        if netG_path != '': # load checkpoint if needed
            self.netG.load_state_dict(torch.load(netG_path))
        print(self.netG)

        self.netD = nn.Sequential(
            nn.Linear(self.x_dim + self.attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        ).to(self.device)

        if netD_path != '':
            self.netD.load_state_dict(torch.load(netD_path))
        print(self.netD)

        # setup optimizer
        if self.adam:
            self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lrD, betas=(self.beta1, 0.999))
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(self.beta1, 0.999))
        else:
            self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
            self.optimizerG = optim.RMSprop(self.netG.parameters(), lr = self.lrG)

    def train(self, train_X, train_attr, train_y):
            
        train_dataset = torch.utils.data.TensorDataset(train_X, train_attr, train_y)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchSize, shuffle=True)
        g_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchSize, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True))
        g_data_iter = iter(g_dataloader)
        def get_attr():
            nonlocal g_data_iter
            try:
                _, attr, y = g_data_iter.next()
            except StopIteration:
                g_data_iter = iter(g_dataloader)
                _, attr, y = g_data_iter.next()
            return attr, y

        input = torch.FloatTensor(self.batchSize, self.x_dim).to(self.device)
        noise = torch.FloatTensor(self.batchSize, self.nz).to(self.device)
        fixed_noise = torch.FloatTensor(self.batchSize, self.nz).normal_(0, 1).to(self.device)

        gen_iterations = 0
        for epoch in range(self.niter):
            data_iter = iter(dataloader)
            i = 0
            while i < len(dataloader):
                ############################
                # (1) Update D network
                ###########################
                for p in self.netD.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update

                # train the discriminator Diters times
                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    Diters = 100
                else:
                    Diters = self.Diters
                j = 0
                while j < Diters and i < len(dataloader):
                    j += 1

                    # clamp parameters to a cube
                    for p in self.netD.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)

                    data = data_iter.next()
                    i += 1

                    # train with real
                    real_X, real_attr, _ = data
                    real_cpu = torch.cat((real_X.to(self.device), real_attr.to(self.device)), dim=-1)
                    self.netD.zero_grad()
                    batch_size = real_cpu.size(0)

                    #if self.cuda:
                    #    real_cpu = real_cpu.cuda()
                    input.resize_as_(real_cpu).copy_(real_cpu)
                    inputv = Variable(input)
                    errD_real = self.netD(inputv)
                    errD_real.backward(torch.ones_like(errD_real))

                    # train with fake
                    noise.resize_(batch_size, self.nz).normal_(0, 1)
                    z_inp = torch.cat((noise.to(self.device), real_attr.to(self.device)), dim=-1)                    
                    noisev = Variable(z_inp)
                    # totally freeze netG
                    with torch.no_grad():
                        fake = Variable(self.netG(noisev).data)
                    inputv = torch.cat((fake, real_attr.to(self.device)), dim=-1)
                    errD_fake = self.netD(inputv)
                    errD_fake.backward(-torch.ones_like(errD_fake))
                    errD = errD_real - errD_fake
                    self.optimizerD.step()

                ############################
                # (2) Update G network
                ###########################
                for p in self.netD.parameters():
                    p.requires_grad = False # to avoid computation
                self.netG.zero_grad()
                # in case our last batch was the tail batch of the dataloader,
                # make sure we feed a full batch of noise
                torch.autograd.set_detect_anomaly(True)
                attr, label_idx = get_attr()
                noise.resize_(attr.shape[0], self.nz).normal_(0, 1)
                z_inp = torch.cat((noise.to(self.device), attr.to(self.device)), dim=-1)   
                noisev = Variable(z_inp)
                fake = self.netG(noisev)
                fake_inp = torch.cat((fake, attr.to(self.device)), dim=-1)   
                errG = self.netD(fake_inp)

                # Doesn't work...
                if False:
                    self.classifier.eval()
                    Y_pred = torch.nn.functional.softmax(self.classifier(fake.detach()), dim=-1)
                    log_prob = torch.log(torch.gather(Y_pred, 1, label_idx.unsqueeze(1)))
                    L_cls = -1 * log_prob # torch.mean(log_prob)
                    errG += self.beta * L_cls

                errG.backward(torch.ones_like(errG))
                self.optimizerG.step()
                gen_iterations += 1

                print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                    % (epoch, self.niter, i, len(dataloader), gen_iterations,
                    errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
                """
                if gen_iterations % 500 == 0:
                    real_cpu = real_cpu.mul(0.5).add(0.5)
                    vutils.save_image(real_cpu, '{0}/real_samples.png'.format(self.experiment))
                    fake = self.netG(Variable(fixed_noise, volatile=True))
                    fake.data = fake.data.mul(0.5).add(0.5)
                    vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(self.experiment, gen_iterations))
                """

            # do checkpointing
            torch.save(self.netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(self.experiment, epoch))
            torch.save(self.netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(self.experiment, epoch))


    def generate_data(self, classes, attributes, n_examples=400):
        '''
        Creates a synthetic dataset based on attribute vectors of unseen class
        Args:
            classes: list of class indices to generate
            attributes: A np array containing class attributes for each class
                of dataset
            n_samples: Number of samples of each unseen class to be generated(Default: 400)
        Returns:
            X and y
        '''
        with torch.no_grad():
            y = torch.tile(classes, (n_examples,)).to(self.device)
            attr = torch.index_select(torch.FloatTensor(attributes).to(self.device), 0, y)
            z = self.Z_dist.sample((len(classes) * n_examples, self.nz)).to(self.device)
            z_inp = torch.cat((z, attr), dim=-1)
            X_gen = self.netG(z_inp).detach()

            assert X_gen.shape[0] == len(classes) * n_examples

            return X_gen, y