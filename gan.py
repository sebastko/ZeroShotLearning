import torch
from torch._C import device
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import uniform, normal

import os
import numpy as np
from sklearn.metrics import accuracy_score


class GanTrainer:
    def __init__(
        self, device, x_dim, z_dim, attr_dim, classifier, **kwargs):
        '''
        Trainer class.
        Args:
            device: CPU/GPU
            x_dim: Dimension of image feature vector
            z_dim: Dimension of noise vector
            attr_dim: Dimension of attribute vector
            kwargs
        '''
        self.device = device

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.attr_dim = attr_dim
        self.classifier = classifier

        self.n_critic = kwargs.get('n_critic', 5)
        self.lmbda = kwargs.get('lmbda', 10.0)
        self.beta = kwargs.get('beta', 0.01)
        #self.bs = kwargs.get('batch_size', 32)

        self.n_train = kwargs.get('n_train')
        self.n_test = kwargs.get('n_test')

        self.eps_dist = uniform.Uniform(0, 1)
        self.Z_dist = normal.Normal(0, 1)

        #self.eps_shape = torch.Size([self.bs, 1])
        #self.z_shape = torch.Size([self.bs, self.z_dim])

        self.net_G = nn.Sequential(
            nn.Linear(self.z_dim + self.attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, self.x_dim),
            nn.ReLU()
        ).to(self.device)
        self.optim_G = optim.Adam(self.net_G.parameters(), lr=1e-4)

        self.net_D = nn.Sequential(
            nn.Linear(self.x_dim + self.attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        ).to(self.device)
        self.optim_D = optim.Adam(self.net_D.parameters(), lr=1e-4)

        self.model_save_dir = "saved_models"
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

    def get_conditional_input(self, X, C_Y):
        new_X = torch.cat([X, C_Y], dim=1).float()
        return autograd.Variable(new_X).to(self.device)

    def get_gradient_penalty(self, X_real, X_gen):
        eps_shape = (X_real.shape[0], 1)
        eps = self.eps_dist.sample(eps_shape).to(self.device)
        X_penalty = eps * X_real + (1 - eps) * X_gen

        X_penalty = autograd.Variable(X_penalty, requires_grad=True).to(self.device)
        critic_pred = self.net_D(X_penalty)
        grad_outputs = torch.ones(critic_pred.size()).to(self.device)
        gradients = autograd.grad(
                outputs=critic_pred, inputs=X_penalty,
                grad_outputs=grad_outputs,
                create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def fit_GAN(self, img_features, label_attr, label_idx, use_cls_loss=True):
        L_gen = 0
        L_disc = 0
        total_L_disc = 0

        img_features = autograd.Variable(img_features.float()).to(self.device)
        label_attr = autograd.Variable(label_attr.float()).to(self.device)
        label_idx = label_idx.to(self.device)

        # =============================================================
        # optimize discriminator
        # =============================================================
        X_real = self.get_conditional_input(img_features, label_attr)
        z_shape = (X_real.shape[0], self.z_dim)
        for _ in range(self.n_critic):
            Z = self.Z_dist.sample(z_shape).to(self.device)
            Z = self.get_conditional_input(Z, label_attr)

            X_gen = self.net_G(Z)
            X_gen = self.get_conditional_input(X_gen, label_attr)

            # calculate normal GAN loss
            L_disc = (self.net_D(X_gen) - self.net_D(X_real)).mean()

            # calculate gradient penalty
            grad_penalty = self.get_gradient_penalty(X_real, X_gen)
            L_disc += self.lmbda * grad_penalty

            # update critic params
            self.optim_D.zero_grad()
            L_disc.backward()
            self.optim_D.step()

            total_L_disc += L_disc.item()

        # =============================================================
        # optimize generator
        # =============================================================
        Z = self.Z_dist.sample(z_shape).to(self.device)
        Z = self.get_conditional_input(Z, label_attr)

        X_gen = self.net_G(Z)
        X = torch.cat([X_gen, label_attr], dim=1).float()
        L_gen = -1 * torch.mean(self.net_D(X))

        if use_cls_loss:
            self.classifier.eval()
            Y_pred = F.softmax(self.classifier(X_gen), dim=0)
            log_prob = torch.log(torch.gather(Y_pred, 1, label_idx.unsqueeze(1)))
            L_cls = -1 * torch.mean(log_prob)
            L_gen += self.beta * L_cls

        self.optim_G.zero_grad()
        L_gen.backward()
        self.optim_G.step()

        return total_L_disc, L_gen.item()
    
    """
    def generate_data(self, labels, attributes, num_examples):
        z_shape = (num_examples, self.z_dim)
        Z = self.Z_dist.sample(z_shape).to(self.device)
        label_attr = attributes[labels]
        Z = self.get_conditional_input(Z, label_attr)
    """

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
        self.net_G.eval()

        y = np.tile(classes, n_examples)
        attr = torch.FloatTensor(attributes[y], device=self.device)
        z = self.Z_dist.sample(torch.Size((len(classes) * n_examples, self.z_dim)))
        z_inp = self.get_conditional_input(z, attr)
        X_gen = self.net_G(z_inp).detach()

        assert X_gen.shape[0] == len(classes) * n_examples

        return X_gen, torch.LongTensor(y, device=self.device)

    def save_model(self):
        g_ckpt_path = os.path.join(self.model_save_dir, "netG.pth")
        torch.save(self.net_G.state_dict(), g_ckpt_path)

        d_ckpt_path = os.path.join(self.model_save_dir, "netD.pth")
        torch.save(self.net_D.state_dict(), d_ckpt_path)

    def load_model(self):
        f1, f2 = False, False
        g_ckpt_path = os.path.join(self.model_save_dir, "netG.pth")
        if os.path.exists(g_ckpt_path):
            self.net_G.load_state_dict(torch.load(g_ckpt_path))
            f1 = True

        d_ckpt_path = os.path.join(self.model_save_dir, "netD.pth")
        if os.path.exists(d_ckpt_path):
            self.net_D.load_state_dict(torch.load(d_ckpt_path))
            f2 = True

        return f1 and f2