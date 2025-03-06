import torch
from gcn.model import ConvolutionalGraphNetwork


class GcnEncoder(ConvolutionalGraphNetwork):
    def __init__(self, input_dimension, hidden_dimension, n_clases, n_layers, dropout=None):
        super().__init__(input_dimension, hidden_dimension, n_clases, n_layers, dropout)