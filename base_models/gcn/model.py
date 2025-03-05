import torch
from layers import GraphConvolution


class ConvolutionalGraphNetwork(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, n_clases, n_layers, dropout = None):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.outpu_dimension = n_clases

        self.dropout = dropout if dropout else 0

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GraphConvolution(self.hidden_dimension, self.hidden_dimension))
        
        self.gcn1 = GraphConvolution(self.input_dimension, self.hidden_dimension)
        self.gcn2 = GraphConvolution(self.hidden_dimension, self.outpu_dimension)
        self.dropout_layer = torch.nn.Dropout(p = self.dropout)

    def forward(self, input, normalized_adjacency):

        # print(input.shape, normalized_adjacency.shape)

        input = self.dropout_layer(input)
        support = torch.nn.functional.relu(self.gcn1(input, normalized_adjacency))
        for layer in self.layers:
            support = layer(support, normalized_adjacency)
        output = self.gcn2(support, normalized_adjacency)
        return torch.nn.functional.softmax(output), support