import torch
from base_models.gcn.layers import GraphConvolution
import numpy as np

class ConvolutionalGraphNetwork(torch.nn.Module):
    def __init__(self, input_dimension, arguments, dropout = None):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = arguments.hidden_dimension
        self.output_dimension = arguments.number_of_classes
        self.embedding_dimension = arguments.embedding_dimension
        self.pred_hidden_dimension = arguments.pred_hidden_dimension
        self.n_embedding_layers = arguments.n_embedding_layers
        self.n_prediction_layers = arguments.n_prediction_layers

        self.bias = True
        self.concat = False

        self.dropout = dropout if dropout else 0

        # self.layers = torch.nn.ModuleList()
        
        self.embedding_first, self.embedding_block,self. embedding_final = self.build_convolutions(True, True)
        self.pred_first, self.pred_block, self.pred_final = self.build_convolutions(True, True, True)

        self.dropout_layer = torch.nn.Dropout(p = self.dropout)


    def build_convolutions(self, add_self, normalize, predict = False):

        
        if predict:
            input_dimension = self.embedding_dimension if not self.concat else self.embedding_dimension * n_layers
            output_dimension = self.output_dimension
            hidden_dimension = self.pred_hidden_dimension
        else:
            input_dimension = self.input_dimension
            output_dimension = self.embedding_dimension
            hidden_dimension = self.hidden_dimension

        n_layers = self.n_prediction_layers if predict else self.n_embedding_layers

        first_conv = GraphConvolution(input_dimension, hidden_dimension, add_self)
        conv_block = torch.nn.ModuleList()

        for _ in range(n_layers):
            conv_block.append(GraphConvolution(
            self.hidden_dimension, self.hidden_dimension, add_self, self.bias, normalize,
        ))
        final_conv = GraphConvolution(self.hidden_dimension, output_dimension, add_self)

        if self.concat:
            final_conv = GraphConvolution(self.hidden_dimension * self.n_prediction_layers, output_dimension, add_self)

        return first_conv, conv_block, final_conv
    
    

    def predict(self, embeddings, normalized_adjacency):
        embeddings = self.dropout_layer(embeddings)
        support = torch.nn.functional.relu(self.pred_first(embeddings, normalized_adjacency))
    
        self.out = list()
        for layer in self.pred_block:
            support = layer(support, normalized_adjacency)
            self.out.append(support)
        if self.concat:
            support = np.hstack([tensor.numpy() for tensor in self.out])

        predictions = self.pred_final(support, normalized_adjacency)
        return torch.nn.functional.softmax(predictions)

    def forward(self, input, normalized_adjacency):
        input = self.dropout_layer(input)
        support = torch.nn.functional.relu(self.embedding_first(input, normalized_adjacency))
        self.dim_out = list()

        for layer in self.embedding_block:
            support = layer(support, normalized_adjacency)
            if self.concat:
                self.dim_out.append(support)
        output = self.embedding_final(support, normalized_adjacency)

        return torch.nn.functional.softmax(output), support