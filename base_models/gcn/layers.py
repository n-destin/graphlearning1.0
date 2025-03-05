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
        torch.nn.init.xavier_uniform(self.weights, gain=1.414)
        if self.bias:
            torch.nn.init.xavier_uniform(self.bias, gain=1.414)

    
    def forward(self, inputs, normalized_adjacency):
        # print(self.weights.shape, inputs.shape)
        before = torch.matmul(inputs, self.weights) # ()
        returning = torch.spmm(normalized_adjacency, before)
        if self.bias:
            returning += self.bias
        return torch.nn.functional.relu(returning)