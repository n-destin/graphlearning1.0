import torch
from torch import nn
from base_models.gcn.model import GraphConvolution
from base_models.graphsage.utils import choose_aggregator


class GraphSageLayer(GraphConvolution):
    def __init__(self, dimension, bias, pooling_type):
        super().__init__(dimension, dimension, bias)
        self.aggregator = choose_aggregator(pooling_type)
        self.concat_linear = torch.nn.Linear(2*dimension, dimension)
        self.weights = torch.nn.Parameter(torch.FloatTensor(2*dimension, dimension))

    def forward(self, input, adjacency):
        aggregated_features = self.aggregator(input, adjacency)
        concatenated = torch.concatenate(input, aggregated_features, dim=-1)
        output = torch.softmax(torch.matmul(self.weights, concatenated))
        if self.bias:
            return torch.norm(output + self.bias)
        return torch.norm(output)