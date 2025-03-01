import torch
from torch import nn
import numpy as np
from data.dataload import normalize
from gcn.train import create_parser
from model import GraphSage


parser = create_parser()
parser.add_argument('--depth', type=int, default=3, help="number of hidden layers")
arguments = parser.parse_args()

arguments.cuda = torch.cuda.is_available() and not arguments.no_cuda

model = GraphSage(arguments.depth, arguments.hidden_dimension)
optimizer = torch.optim.Adam(model.parameters, lr=arguments.learning_rate, weight_decay=arguments.weight_decay)

features, adjancency = [], torch.eye()

def loss_function(features, adjacency):
    features_transpose = torch.transpose(features)
    correlation = torch.matmul(features_transpose, features)
    adjacency_correlation = torch.matmul(correlation, adjacency)
    row_sum = torch.sum(adjacency_correlation, dim = 1)
    return torch.sum(row_sum)

if arguments.cuda:
    model = model.cuda()
    features = features.cuda()
    adjancency.cuda()

class NeuralAggregation(nn.Module):
    def __init__(self, dimension, bias):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.FloatTensor(dimension, dimension))
        if self.bias:
            self.bias = bias
        else:
            self.register_parameter('bias', None)
        
    def forward(self, features, adjacency):
        output = torch.matmul(features, self.weights)
        if self.bias:
            return max_aggregator(output + self.bias, adjacency)
        return max_aggregator(output, adjacency)

def max_aggregator(features, ajdacency):
    nodes, dimension = features.shape
    returning = torch.zeros(nodes, dimension)
    for node in range(len(features)):
        for neigbor in ajdacency[node]:
            returning[node] = torch.maximum(returning[node], neigbor * features[node])       
    return returning
    
def mean_aggregator(feature_matrix, adjanceny):
    normalized_adjacency = normalize(adjanceny)
    return torch.matmul(normalized_adjacency, feature_matrix) 

def choose_aggregator(pooling_type, dimension = None, bias = None):
    if pooling_type == "mean":
        return mean_aggregator
    neural_aggregator = NeuralAggregation(dimension, bias)
    return neural_aggregator

def train(epoch):
    print("Epoch : {}".format(epoch))
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss = loss_function(output)
    features = output
    print("Loss : {}".format(loss.item()))
    loss.backward()
    optimizer.step()


def test(features):
    model.test()
    output = model(features)
    print('Loss : {}'.format(loss_function(output)))

for epoch in range(arguments.epochs):
    train(epoch)
    
test(features)