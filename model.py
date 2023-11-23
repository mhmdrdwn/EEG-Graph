import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool


def normalize_A(A, lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N)-torch.eye(N,N))
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N)-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N)
    return Lnorm


def normalize_all_A(A):
    all_A = []
    for adj in A:
        all_A.append(normalize_A(adj).detach().numpy())
    all_A = np.array(all_A)
    all_A = torch.tensor(all_A)
    return all_A


class ChebNetConv(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(ChebNetConv, self).__init__()

        self.K = k
        self.linear = nn.Linear(in_features * k, out_features)

    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor):
        x = self.__transform_to_chebyshev(x, laplacian)
        x = self.linear(x)
        
        return x

    def __transform_to_chebyshev(self, x, laplacian):
        cheb_x = x.unsqueeze(2)
        x0 = x
        if self.K > 1:
            x1 = torch.matmul(laplacian, x0)
            cheb_x = torch.cat((cheb_x, x1.unsqueeze(2)), 2)
            for _ in range(2, self.K):
                x2 = 2 * torch.matmul(laplacian, x1) - x0
                cheb_x = torch.cat((cheb_x, x2.unsqueeze(2)), 2)
                x0, x1 = x1, x2
        cheb_x = cheb_x.reshape(cheb_x.shape[0], cheb_x.shape[1], cheb_x.shape[2]*cheb_x.shape[3])
        return cheb_x
import torch.nn.functional as F


class ChebNetGCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_electrodes, out_channels, 
                 num_classes, num_hidden_layers=2, dropout=0, residual=False, k=2):
        super(ChebNetGCN, self).__init__()

        self.dropout = dropout
        self.residual = residual

        self.input_conv = ChebNetConv(input_size, hidden_size, k)
        self.hidden_convs = nn.ModuleList([ChebNetConv(hidden_size, hidden_size, k) for _ in range(num_hidden_layers)])
        self.output_conv = ChebNetConv(hidden_size, out_channels, k)
        self.BN1 = nn.BatchNorm1d(out_channels)
        self.fc = nn.Linear(out_channels, num_classes)
        
    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor, edge_weight=None):
        laplacian = normalize_all_A(laplacian)
        if edge_weight is not None:
            laplacian = edge_weight * laplacian
        x = F.dropout(x, p=self.dropout, training=self.training)
        #print(x.shape)
        x = F.relu(self.input_conv(x, laplacian))
        #print(x.shape)
        for conv in self.hidden_convs:
            if self.residual:
                x = F.relu(conv(x, laplacian)) + x
            else:
                x = F.relu(conv(x, laplacian))
        #print(x.shape)        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_conv(x, laplacian)
        #print(x.shape)
        x = x.squeeze()
        batch =None
        x = global_mean_pool(x, batch)
        #print(x.shape)
        x = self.BN1(x)
        x = self.fc(x)
        #print(x.shape)
        return x