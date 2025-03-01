import torch
from torch import nn
from layers import GraphSageLayer

class GraphSage(nn.Module):
    def __init__(self, depth, dimension, pooling_type):
        super().__init__()
        self.depth = depth
        self.dimension = dimension
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth):
            self.hidden_layers.append(GraphSageLayer(dimension, None, pooling_type))
    
    def forward(self, inputs):
        support = inputs 
        for layer in self.hidden_layers:
            support = layer(support)
        return support
    