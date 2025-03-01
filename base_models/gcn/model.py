import torch
from base_models.gcn.layers import GraphConvolution


class ConvolutionalGraphNetwork(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, n_clases, n_layers):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.outpu_dimension = n_clases
        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GraphConvolution(self.hidden_dimension, self.hidden_dimension))
        self.gcn1 = GraphConvolution(self.input_dimension, self.hidden_dimension)
        self.gcn2 = GraphConvolution(self.hidden_dimension, self.outpu_dimension)

    def forward(self, input, normalized_adjacency):
        support = self.gcn1(input, normalized_adjacency)
        for layer in self.layers:
            support = self.layer(support, normalized_adjacency)
        output = self.gcn2(support, normalized_adjacency)
        return output