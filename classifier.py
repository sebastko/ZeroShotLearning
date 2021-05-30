import torch
from torch import nn
import copy


class MaskedClassifier(nn.Module):
    def __init__(self, x_dim, mask, device):
        super().__init__()
        self.x_dim = x_dim
        self.mask = mask.to(device)

        self.all_classes_dim, = mask.shape
        self.net = nn.Sequential(nn.Linear(self.x_dim, self.all_classes_dim)).to(device)

    def forward(self, X):
        pred = self.net(X)
        masked_pred = torch.where(self.mask, pred, torch.FloatTensor([-1e9]))
        return masked_pred


def train_cls(cls_mask, train_X, train_y, val_X, val_y, device):
    _, fea_dim = train_X.shape
    linear_cls = MaskedClassifier(fea_dim, cls_mask, device)
    optimizer = torch.optim.Adam(linear_cls.parameters())
    criterion = nn.CrossEntropyLoss()

    def accuracy(pred, y):
        return (torch.sum(torch.argmax(pred, dim=-1) == y)).item() / y.shape[0]

    best_val_acc = 0.0
    best_linear_cls = None
    for i in range(5): # 100
        optimizer.zero_grad()
        train_pred = linear_cls(train_X.to(device))
        loss = criterion(train_pred, train_y)
        loss.backward()
        optimizer.step()
        
        val_seen_pred = linear_cls(val_X.to(device))
        train_acc = accuracy(train_pred, train_y)
        val_acc = accuracy(val_seen_pred, val_y)
        print(f'Loss: {loss}, train acc: {train_acc}, val acc: {val_acc}')
        if val_acc > best_val_acc:
            print('Best!')
            best_val_acc = val_acc
            best_linear_cls = copy.deepcopy(linear_cls).to(device)

    return best_linear_cls
