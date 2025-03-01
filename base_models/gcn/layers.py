import torch
import math

class GraphConvolution(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, bias = 0):
        super(GraphConvolution, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weights = torch.nn.Parameter(torch.FloatTensor(input_dimension, output_dimension))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor.uniform_(output_dimension))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        '''Xavier initialization'''
        dev = 1/math.sqrt(self.output_dimension)
        self.weights.data.uniform_(-dev, dev)
        if self.bias:
            self.bias.data.uniform_(-dev, dev)

    
    def forward(self, inputs, normalized_adjacency):
        before = torch.matmul(inputs, self.weights)
        returning = torch.spmm(before, normalized_adjacency)
        if self.bias:
            returning += self.bias
        return torch.nn.functional.softmax(returning)