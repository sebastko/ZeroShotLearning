import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, x_dim, attr_dim, nz, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, x_dim),
            nn.ReLU()
        ).to(device)

    def forward(self, noise, attr):
        inp = torch.cat((noise, attr), dim=-1)
        return self.net(inp)


class Discriminator(nn.Module):
    def __init__(self, x_dim, attr_dim, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + attr_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, X, attr):
        inp = torch.cat((X, attr), dim=-1)
        return self.net(inp)
