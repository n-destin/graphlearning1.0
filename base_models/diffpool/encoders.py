import torch
import random
from collections import defaultdict
import torch.nn as nn
from base_models.gcn.model import ConvolutionalGraphNetwork
from base_models.graphsage.model import GraphSage
from base_models.data.datasets.data_utils import normalize
import networkx as nx


def choose_model(model_name):
    if model_name == "graphsage":
        return GraphSage
    elif model_name == "gcn":
        return ConvolutionalGraphNetwork
    

def PoolingSelector(arguments, encoder_type = None):
    if encoder_type == "set2set":
        return Set2SetPooling(arguments.input_dimension, arguments.steps)
    elif encoder_type == "meanpooling":
        return MeanPooling(None, arguments) # change the adhancecny being none

    return Diffpool(arguments)

class Set2SetPooling(nn.Module):
    def __init__(self, input_dimension, steps, lstm_n_layers = 1):
        super(Set2SetPooling, self).__init__()
        self.input_dimension = input_dimension
        self.steps = steps
        self.lstm_n_layers = lstm_n_layers
        self.projectionLayer = nn.Linear(self.input_dimension * self.steps, self.input_dimension)
        self.lstm = nn.LSTM(self.input_dimension * 2, self.input_dimension, self.lstm_n_layers, bias = True)

    def forward(self, input):

        hidden_state = torch.zeros(self.lstm_n_layers, 1, self.input_dimension)
        memory = torch.zeros(self.lstm_n_layers, 1, self.input_dimension)
        contexts = torch.Tensor(self.steps, self.input_dimension)

        for _ in range(self.steps):
            scores = nn.functional.softmax(torch.matmul(input, hidden_state.squeeze()))
            context = torch.matmul(scores, input)
            contexts[_, :] = context
            context = context.reshape(1, 1, self.input_dimension)
            _, (hidden_state, memory) = self.lstm(torch.cat((context, hidden_state), dim=2), (hidden_state, memory))

        return nn.functional.softmax(self.projectionLayer(torch.Tensor(contexts).flatten()))
    

class MeanPooling(nn.Module):
    ''''
    Perform one mean message passing (after sampling) and return the mean of the embddings
    '''
    def __init__(self, adjacency, arguments):

        self.adjacency = adjacency
        if arguments.n_sample is not None:
            node_neighbors = defaultdict(int)
            for node, row in enumerate(adjacency):
                for neighbor, connection in enumerate(row):
                    if connection == 1:
                        node_neighbors[node] += [neighbor]
            
            row_indices = list()
            col_indices = list()
            for node, neighbors in node_neighbors.items():
                node_neighbors[node] = random.sample(neighbors, arguments.n_sample)
                for neighbor in node_neighbors[node]:
                    row_indices.append(node)
                    col_indices.append(neighbor)
           
            new_adj = torch.zeros(adjacency.shape)
            new_adj[row_indices, col_indices] = 1
            self.adjacency = new_adj
    
    def forward(self, embeddings):
        normalized_adj  = normalize(self.adjacency)
        node_embedding = torch.matmul(normalized_adj, embeddings)
        
        return torch.mean(node_embedding, dim = 0)


class Diffpool(nn.Module):
    def __init__(self, arguments):
        super(Diffpool, self).__init__()
        self.n_layers = arguments.n_layers
        self.embedding_models = list()
        self.pooling_models = list()

        for index, _ in enumerate(self.n_layers):
            if arguments.embedding_model[index] == "graphsage":
                self.embedding_models.append(GraphSage(arguments.graph_sage_depth, arguments.input_dimension, 'neural'))
            else:
                self.embedding_models.append(ConvolutionalGraphNetwork(arguments.input_dimension, arguments))
            
            if arguments.pooling_models[index] == 'graphsage':
                self.pooling_models.append(GraphSage(arguments.graph_sage_depth, arguments.input_dimensions, 'nueral'))
            else:
                self.pooling_models.append(ConvolutionalGraphNetwork(arguments.input_dimension, arguments))
    
    def forward(self, input, adjacency):
        cluster_features = input
        for index in range(len(self.n_layers)):
            cluster_embeddings, assignment_matrix = self.embedding_models[index](cluster_features, adjacency), self.pooling_models(cluster_features, adjacency)
            cluster_features = torch.mm(assignment_matrix, cluster_embeddings)
            adjacency = torch.mm(adjacency, cluster_embeddings)
        

        return torch.mean(cluster_features, dim=0)
            


class Encoder(nn.Module):
    def __init__(self, graphs, arguments):
        self.graphs = graphs
        self.input_dimension = arguments.input_dimension
        self.encoder = PoolingSelector(arguments.encoder_type)
        self.node_embeddings = ConvolutionalGraphNetwork(arguments.input_dimension, arguments)
        self.first_conv, self.conv_block, self.last_conv = self.node_embeddings.build_convolutions(True, True, True)
        self.prediction_block = nn.ModuleList()
        self.prediction_block.append(nn.Linear(arguments.input_dimension, arguments.pred_hidden_dimension))
        
        for _ in range(len(arguments.pred_n_layers)):
            self.prediction_block.append(nn.Linear(arguments.pred_hidden_dimension, arguments.pred_hidden_dimension))
        self.prediction_block.append(nn.Linear(arguments.pred_hidden_dimension, self.output_dimension))
        
    def forward(self, graphs):
        predictions = torch.FloatTensor(len(self.graphs), self.output_dimension)
        for index, graph in enumerate(graphs):
            normalized_adjancency = normalize(nx.adjacency_matrix(graph))
            features = torch.FloatTensor(len(nodes), self.input_dimension)
            nodes = graph.nodes()

            for index, node in enumerate(nodes):
               features[index, :] = node['feature']
            
            embeddings = self.node_embeddings(features, normalized_adjancency)
            graph_embedding = self.encoder(embeddings)
            predictions[index] = self.prediction_block(graph_embedding)
        
        return predictions
    

    ''''

# Testing Set2Set Pooling 


Pooling = Set2SetPooling(arguments.embedding_dimension, 2)
embeddings = torch.zeros(100, arguments.embedding_dimension)

with open(embeddings_pathname, "r") as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        embedding  = torch.FloatTensor([float(num) for num in line.split(' ')])
        embeddings[index, :] = embedding 

print(Pooling(embeddings, ))
'''