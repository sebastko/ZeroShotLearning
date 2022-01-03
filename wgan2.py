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
from torch import autograd
from torch.distributions import uniform, normal
import torch.nn.functional as F
import os
import json
import models

class CLSWGAN:
    eps = uniform.Uniform(0, 1)
    Z_sampler = normal.Normal(0, 1)

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

        self.netG = models.Generator(x_dim, attr_dim, self.nz, device)

        self.netG.apply(weights_init)
        if netG_path != '': # load checkpoint if needed
            self.netG.load_state_dict(torch.load(netG_path))
        print(self.netG)

        self.netD = models.Discriminator(x_dim, attr_dim, device)

        if netD_path != '':
            self.netD.load_state_dict(torch.load(netD_path))
        print(self.netD)

        self.G_cls_criterion = nn.NLLLoss()

        # setup optimizer
        if self.adam:
            self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lrD, betas=(self.beta1, 0.999))
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(self.beta1, 0.999))
        else:
            self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lrD)
            self.optimizerG = optim.RMSprop(self.netG.parameters(), lr = self.lrG)

    def get_noise(self, batch_size):
        return torch.autograd.Variable(self.Z_sampler.sample(torch.Size([batch_size, self.nz])).to(self.device))

    def get_gradient_penalty(self, d_real, d_fake, batch_size, atts):
        eps = self.eps.sample(torch.Size([batch_size, 1])).to(self.device)
        X_penalty = eps * d_real + (1 - eps) * d_fake

        X_penalty = autograd.Variable(X_penalty, requires_grad=True).to(self.device)
        d_pred = self.netD(X_penalty, atts)
        grad_outputs = torch.ones(d_pred.size()).to(self.device)
        gradients = autograd.grad(
            outputs=d_pred, inputs=X_penalty,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lmbda
        return grad_penalty

    def step_wgan(self, **params):
        loss_g = None
        self.classifier.eval()
        self.netG.eval()
        for p in self.netD.parameters():
            p.requires_grad = True
        #for p in self.netG.parameters():
        #    p.requires_grad = False

        batch_size = params['atts'].shape[0]
        self.optimizerD.zero_grad()
        d_real = self.netD(params['feat'], params['atts'])
        d_real = torch.mean(d_real)
        d_real.backward(torch.tensor(-1.))

        Z = self.get_noise(batch_size)
        fake_feat = self.netG(Z, params['atts'])

        d_fake = self.netD(fake_feat, params['atts'])
        d_fake = torch.mean(d_fake)
        d_fake.backward(torch.tensor(1.))

        gradient_penalty = self.get_gradient_penalty(params['feat'], fake_feat, batch_size, params['atts'])
        gradient_penalty.backward()

        loss_d = d_fake - d_real + gradient_penalty
        self.optimizerD.step()

        if params['step'] % self.Diters == 0:
            for p in self.netD.parameters():
                p.requires_grad = False
            self.netG.train()
            #for p in self.netG.parameters():
            #    p.requires_grad = True
            self.optimizerG.zero_grad()
            Z = self.get_noise(batch_size)
            fake_feat = self.netG(Z, params['atts'])

            d_fake = self.netD(fake_feat, params['atts'])
            d_fake = -1 * torch.mean(d_fake)

            g_cls_pred = self.classifier(fake_feat)
            loss_cls = self.G_cls_criterion(F.log_softmax(g_cls_pred, dim=1), params['cls_true'])
            loss_g = d_fake + self.beta * loss_cls

            loss_g.backward()
            self.optimizerG.step()

        return loss_d.item(), loss_g.item() if loss_g is not None else 0


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
            X_gen = self.netG(z, attr).detach()

            assert X_gen.shape[0] == len(classes) * n_examples

            return X_gen, y