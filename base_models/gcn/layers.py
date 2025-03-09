import torch
import math

class GraphConvolution(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, add_self, bias = None, normalize = False):
        super(GraphConvolution, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.add_self = add_self
        self.normalize = normalize
        self.weights = torch.nn.Parameter(torch.FloatTensor(input_dimension, output_dimension))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(output_dimension).uniform_())
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        '''Xavier initialization'''
        torch.nn.init.xavier_uniform(self.weights, gain=1.414)
        if self.bias != None:
            torch.nn.init.normal_(self.bias)

    
    def forward(self, inputs, normalized_adjacency):
        # print(inputs.shape, self.weights.shape)
        # inputs : num_nodes x dimension weighs. weighs, dimension weights x output_dimension -> num_nodes x output_dim
        before = torch.matmul(normalized_adjacency, inputs) # ()
        if self.add_self:
            before += inputs
        returning = torch.spmm(before, self.weights)
        if self.bias != None:
            returning += self.bias
        
        return torch.nn.functional.relu(returning)