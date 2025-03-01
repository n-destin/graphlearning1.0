# -*- coding: utf-8 -*-
"""GATLayer.ipynb
"""

import os
import torch

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import math
import numpy as np
import scipy.sparse as sp

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.
edge_index = data.edge_index
adjacency_matrix = np.zeros((data.num_nodes, data.num_nodes))
adjacency_matrix[edge_index[0], edge_index[1]] = 1
adjacency_matrix = torch.FloatTensor(adjacency_matrix)

device = ('cuda' if torch.cuda.is_available() else "cpu")

class GATLayer(nn.Module):
  '''
  ### GAT layer with one attention head, check the original paper: https://arxiv.org/pdf/1710.10903.pdf
  '''
  def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, num_heads=2, concat=True):
    super(GATLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features 
    self.dropout = dropout 
    self.alpha = alpha 

    self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
    nn.init.xavier_uniform_(self.W.data, gain=1.414)

    self.a = nn.Parameter(torch.FloatTensor(num_heads, 2*out_features, 1))
    nn.init.xavier_uniform_(self.a.data, gain=1.414)

    self.leaky_relu = nn.LeakyReLU(self.alpha)
    self.dropout_layer = nn.Dropout(self.dropout)

    self.num_heads = num_heads

    self.concat = concat
  def forward(self, x, adj):
        N = x.size(0)
        adj = adj + torch.eye(N, N)
        x_transformed = torch.bmm(torch.unsqueeze(x, 0).repeat(self.num_heads, 1, 1), self.W)

        f_repeat = x_transformed.repeat_interleave(N, dim=1)
        f_repeat_interleave = x_transformed.repeat(1, N, 1)
        all_features = torch.cat([f_repeat, f_repeat_interleave], dim=-1)

        attention_scores = self.leaky_relu(torch.matmul(all_features, self.a).squeeze(-1))
        attention_scores = attention_scores.view(self.num_heads, N, N)
        zero_vec = -9e15*torch.ones_like(attention_scores)

        attention_scores = torch.where(adj > 0, attention_scores, zero_vec)
        attention_scores_normalized = F.softmax(attention_scores, dim=-1)
        attention_scores_normalized = self.dropout_layer(attention_scores_normalized)
        h_prime = torch.matmul(attention_scores_normalized, x_transformed)

        if self.concat:
            h_prime = F.elu(h_prime.view(N, self.num_heads * self.out_features))
        else:
            h_prime = F.elu(h_prime.mean(dim=0))

        return h_prime

class GAT(nn.Module):
  """
    GAT model
  """
  def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.6, alpha=0.2, num_heads=2):
    super(GAT, self).__init__()
    self.dropout = dropout 
    self.first = GATLayer(nfeat, nhid, dropout, alpha, num_heads, True)
    self.last = GATLayer(nhid*num_heads, nclass,dropout, alpha, num_heads, False)
    self.attentions = [GATLayer(nhid*num_heads, nhid, dropout, alpha, num_heads, True) for _ in range(nlayers-2)]

  def forward(self, x, adj):
    returned = self.first(x, adj)
    for layer in self.attentions:
      returned = layer(returned, adj)

    returned = self.last(returned, adj)
    return returned

model = GAT(dataset.num_features, 16, dataset.num_classes)
criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
def train_gat():
    model.train()
    optimizer.zero_grad() 
    out = model(data.x, adjacency_matrix) 
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
    loss.backward() 
    optimizer.step() 
    return loss

def test_gat():
    model = GAT(dataset.num_features, 16, dataset.num_classes)
    model.load_state_dict("model.pt")
    model.eval()
    out = model(data.x, adjacency_matrix)
    pred = out.argmax(dim=1) 
    test_correct = pred[data.test_mask] == data.y[data.test_mask] 
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
    return test_acc

from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

least_loss = 2e2 # a random number

for epoch in range(1, 201):
    loss = train_gat()
    if loss<least_loss:
      torch.save(model.state_dict, "model.pt")
      least_loss = loss
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test_gat()
print(f'Test Accuracy: {test_acc:.4f}')