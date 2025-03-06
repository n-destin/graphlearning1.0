import torch
from layers import GraphConvolution


class ConvolutionalGraphNetwork(torch.nn.Module):
    def __init__(self, input_dimension, hidden_dimension, embedding_dimension, n_clases, n_layers, dropout = None):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = n_clases
        self.embedding_dimension = embedding_dimension

        self.bias = True
        self.concat = False

        self.dropout = dropout if dropout else 0

        self.layers = torch.nn.ModuleList()
        
        self.embedding_first, self.embedding_block,self. embedding_final = self.build_convolutions(True, False, True, n_layers, False)
        self.pred_first, self.pred_block, self.pred_final = self.build_convolutions(True, False, True, 2, True)

        self.dropout_layer = torch.nn.Dropout(p = self.dropout)


    def build_convolutions(self, add_self, bn, normalize, n_layers, predicit = False):
        
        if predicit:
            input_dimension = self.embedding_dimension
            output_dimension = self.output_dimension
            hidden_dimension = self.pred_hidden_dimension
        else:
            input_dimension = self.input_dimension
            output_dimension = self.embedding_dimension
            hidden_dimension = self.hidden_dimension

        first_conv = GraphConvolution(input_dimension, hidden_dimension)
        conv_block = torch.nn.ModuleList([GraphConvolution(
            self.hidden_dimension, self.hidden_dimension, add_self, bn, self.bias, normalize, self.dropout,
        )] for _ in range(len(n_layers)))
        embedding_conv = GraphConvolution(self.hidden_dimension, output_dimension)

        return first_conv, conv_block, embedding_conv
    
        

    def forward(self, input, normalized_adjacency):

        # print(input.shape, normalized_adjacency.shape)

        input = self.dropout_layer(input)
        support = torch.nn.functional.relu(self.embedding_first(input, normalized_adjacency))
        self.dim_out = list()

        for layer in self.embedding_block:
            if self.concat:
                self.dim_out.append(support)
            support = layer(support, normalized_adjacency)
        output = self.embedding_final(support, normalized_adjacency)

        # create 


        return torch.nn.functional.softmax(output), support