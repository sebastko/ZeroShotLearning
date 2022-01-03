import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import copy
from tqdm import tqdm, trange

import dataset
from gan import GanTrainer
from wgan import WassersteinGAN
from wgan2 import CLSWGAN
from classifier import MaskedClassifier, train_cls
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import uniform, normal

def get_dataset_and_print(data, type, device):
    X, y, classes, mask = data.get_dataset(type, device)
    print('---------------')
    print(f'Data set {type}:')
    print(f'  N:          {X.shape[0]}')
    print(f'  classes:    {classes}')
    print(f'  num of cls: {len(classes)}')
    print(f'  mask:       {mask}')
    return X, y, classes, mask

def cli():
    #parser.add_argument('--cuda'  , action='store_true', help='enables cuda')

    """
    os.system('mkdir {0}'.format(self.experiment))

    self.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", self.manualSeed)
    random.seed(self.manualSeed)
    torch.manual_seed(self.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not self.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    data_features_dir = '/mnt/c/datasets/xlsa17/data/AWA2'
    data = dataset.ParentDataset(data_features_dir)

    total_N, _ = data.features.shape
    train_all_X, train_all_y, train_all_classes, train_all_mask = get_dataset_and_print(data, 'train', device)
    train_all_N, fea_dim = train_all_X.shape

    # TODO: make gin configurable?
    seen_val_ratio = 0.2
    
    train_N = int(train_all_X.shape[0] * (1-seen_val_ratio))
    train_X, train_y, train_mask = train_all_X[:train_N, :], train_all_y[:train_N], train_all_mask
    val_seen_X, val_seen_y, val_seen_mask = train_all_X[train_N:, :], train_all_y[train_N:], train_all_mask

    val_unseen_X, val_unseen_y, val_unseen_classes, val_unseen_mask = get_dataset_and_print(data, 'val', device)
    test_seen_X, test_seen_y, test_seen_classes, test_seen_mask = get_dataset_and_print(data, 'test_seen', device)
    test_unseen_X, test_unseen_y, test_unseen_classes, test_unseen_mask = get_dataset_and_print(data, 'test_unseen', device)

    assert total_N == train_all_X.shape[0] + val_unseen_X.shape[0] + test_seen_X.shape[0] + test_unseen_X.shape[0]

    # 1. Train a linear softmax classifier P(y|x;θ) on the real features of seen classes.
    print('1. Train a linear softmax classifier P(y|x;θ) on the real features of seen classes.')
    best_linear_cls, linear_cls_val_acc = train_cls(train_mask, train_X, train_y, val_seen_X, val_seen_y, device)

    # 2. Train  the  conditional f-CLSWGAN generator, conditioned on class attributes a_y
    print('2. Train  the  conditional f-CLSWGAN generator, conditioned on class attributes a_y')

    attr_dim = data.attributes.shape[-1]
    train_attr = torch.FloatTensor(data.attributes[train_y.cpu().detach().numpy()])
    assert train_attr.shape == (train_X.shape[0], attr_dim)

    #gan_trainer = GanTrainer(device, fea_dim, attr_dim, attr_dim, classifier=best_linear_cls)
    #gan_trainer = WassersteinGAN(device, fea_dim, attr_dim, attr_dim, classifier=best_linear_cls, niter=10)
    gan_trainer = CLSWGAN(device, fea_dim, attr_dim, attr_dim, classifier=best_linear_cls, niter=10, beta=0.1)


    n_examples_per_class = train_X.shape[0] // len(val_unseen_classes)
    synth_y = torch.tile(val_unseen_classes, (n_examples_per_class,)).to(device)
    synth_attr = torch.index_select(torch.FloatTensor(data.attributes).to(device), 0, synth_y)
    fixed_noise = normal.Normal(0, 1).sample((len(val_unseen_classes) * n_examples_per_class, attr_dim)).to(device)
    
    n_epochs = 100
    for ep in trange(1, n_epochs + 1):
        d_x, d_g_z = 0, 0
        #orig_g_params = copy(gan_trainer.netG.parameters())
        loss_dis, loss_gen = gan_trainer.step_wgan(feat=train_X.to(device), atts=train_attr.to(device), cls_true=train_y.to(device), step=ep)
        print("Loss for epoch: %3d - D: %.4f | G: %.4f | D(x) acc: %.4f | D(G(z)) acc: %.4f"\
                %(ep, loss_dis, loss_gen, d_x, d_g_z))
        
        #new_g_params = gan_trainer.netG.parameters()
    
        # 3. Augment the training set by generating synthetic examples of unseen classes using generator G.
        print('  - generate synthetic examples')

        n_examples_per_class = train_X.shape[0] // len(val_unseen_classes)

        synth_X = gan_trainer.netG(fixed_noise, synth_attr).detach()

        assert train_X.shape[-1] == synth_X.shape[-1]
        assert len(synth_y.shape) == 1
        assert synth_X.shape[0] == synth_y.shape[0]

        # 4. Train the final classifier.
        print('  - train classifier')

        # ZSL
        #print('==================')
        #print(' ZSL:')
        #print('------------------')
        zsl_cls, zsl_val_acc = train_cls(val_unseen_mask, synth_X, synth_y, val_unseen_X, val_unseen_y, device, verbose=False)
        print(f'     - ZSL val acc: {zsl_val_acc}')

        # GZSL
        #print('==================')
        #print(' GZSL:')
        #print('------------------')
        train_val_mask = train_mask + val_unseen_mask
        gzsl_train_X = torch.cat((train_X, synth_X), dim=0)
        gzsl_train_y = torch.cat((train_y, synth_y), dim=0)
        val_X = torch.cat((val_seen_X, val_unseen_X), dim=0)
        val_y = torch.cat((val_seen_y, val_unseen_y), dim=0)
        gzsl_cls, gzsl_val_acc = train_cls(train_val_mask, gzsl_train_X, gzsl_train_y, val_X, val_y, device, verbose=False)
        print(f'     - GZSL val acc: {gzsl_val_acc}')


if __name__ == "__main__":
    cli()