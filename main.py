import torch
import torch.nn as nn
import numpy as np
import copy

import dataset


def get_dataset_and_print(data, type):
    X, y, mask = data.get_dataset(type)
    print('---------------')
    print(f'Data set {type}:')
    print(f'  N:       {X.shape[0]}')
    print(f'  classes: {int(sum(mask))}')
    print(f'  mask:    {mask}')
    return X, y, mask

def cli():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    data_features_dir = '/mnt/c/datasets/xlsa17/data/AWA2'
    data = dataset.ParentDataset(data_features_dir)

    total_N, _ = data.features.shape
    train_all_X, train_all_y, train_all_mask = get_dataset_and_print(data, 'train')
    train_all_N, D = train_all_X.shape

    # TODO: make gin configurable?
    seen_val_ratio = 0.2
    
    train_N = int(train_all_X.shape[0] * (1-seen_val_ratio))
    train_X, train_y, train_mask = train_all_X[:train_N, :], train_all_y[:train_N], train_all_mask
    val_seen_X, val_seen_y, val_seen_mask = train_all_X[train_N:, :], train_all_y[train_N:], train_all_mask

    val_unseen_X, val_unseen_y, val_unseen_mask = get_dataset_and_print(data, 'val')
    test_seen_X, test_seen_y, test_seen_mask = get_dataset_and_print(data, 'test_seen')
    test_unseen_X, test_unseen_y, test_unseen_mask = get_dataset_and_print(data, 'test_unseen')

    # C - number of all classes (seen and unseen).
    C, = train_mask.shape

    assert total_N == train_all_X.shape[0] + val_unseen_X.shape[0] + test_seen_X.shape[0] + test_unseen_X.shape[0]

    # 1. Train a linear softmax classifier P(y|x;θ) on the real features of seen classes.
    _, D = train_X.shape
    linear_cls = nn.Sequential(nn.Linear(D, C))
    optimizer = torch.optim.Adam(linear_cls.parameters())
    criterion = nn.CrossEntropyLoss()

    def masked_cls(X, mask):
        pred = linear_cls(X)
        masked_pred = torch.where(mask, pred, torch.FloatTensor([-1e9]))
        return masked_pred
    
    def accuracy(pred, y):
        return (torch.sum(torch.argmax(pred, dim=-1) == y)).item() / y.shape[0]

    best_val_acc = 0.0
    best_linear_cls = None
    for i in range(100):
        optimizer.zero_grad()
        train_pred = masked_cls(train_X, train_mask)
        loss = criterion(train_pred, train_y)
        loss.backward()
        optimizer.step()
        
        val_seen_pred = masked_cls(val_seen_X, val_seen_mask)
        train_acc = accuracy(train_pred, train_y)
        val_seen_acc = accuracy(val_seen_pred, val_seen_y)
        print(f'Loss: {loss}, train acc: {train_acc}, val_seen acc: {val_seen_acc}')
        if val_seen_acc > best_val_acc:
            print('Best!')
            best_val_acc = val_seen_acc
            best_linear_cls = copy.deepcopy(linear_cls)

    # 2. Train  the  conditional f-CLSWGAN generator, conditioned on class attributes a_y

    # 3. Augment the training set by generating synthetic examples of unseen classes using generator G.

    # 4. Train the final classifier.


if __name__ == "__main__":
    cli()