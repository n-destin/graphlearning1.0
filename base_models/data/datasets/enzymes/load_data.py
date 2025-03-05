import torch
import numpy as np
from collections import defaultdict
from base_models.data.datasets.data_utils import one_hot_encode_labels
import networkx as nx 
import os 


script_dir = os.path.dirname(os.path.abspath(__file__))

class Dataset():

    '''
    Constructs features from graphs extracted from load_data
    '''

    def __init__(self, graphs, feature_type, max_deg, max_nodes = None):

        self.graphs = graphs
        self.feature_type = feature_type
        
        if not max_nodes:
            for graph in self.graph:
                self.max_nodes = max(self.max_nodes, len(graph.nodes()))
        self.features = []
        self.labels = []
        self.adjacencies = []
        self.num_nodes = []
        self.assigned_dimensions = []
        self.max_deg = max_deg


    def pad(self, matrix):
        return np.pad(matrix, pad_width=self.max_nodes - len(matrix), mode="constant", constant_values=0)
        

    def construct_features(self, ):

        for graph in self.graphs:
            
            adjacency = nx.to_numpy_array(graph)
            n_nodes = graph.nodes()
            def_dimension = graph.nodes()[0]['feature'].shape()[1]
            degrees = np.sum(adjacency, axis=1)
            # one hot encode degrees 
            degrees_ = np.zeros(self.max_nodes, max(degrees) + 1)
            degrees_one_hot = self.pad(degrees_[[i for i in range(len(self.max_nodes))], degrees])
            
            self.adjacencies.append(adjacency)
            self.labels.append(graph['label'])        
            self.num_nodes.append(len(graph.nodes()))
            def_features = np.zeros((n_nodes, def_dimension))

            for node_index, node in graph.nodes():
                def_features[node_index :] = node['feature']
            
            if self.feature_type == "defaault":
                self.features.append(def_features)
            
            elif self.features == 'degree_number':
                self.features.append(np.expand_dims(self.pad(degrees), 0))
            
            elif self.feature_type  == 'degree_and_feature':
                self.features.append(np.hstack(degrees_one_hot, degrees_))
            
            elif self.feature_type == "structure":
                
                clustering = np.expand_dims(self.pad(list(nx.clustering(graph).values())), 0)
                features = np.hstack[features, clustering]
                self.features.append(np.hstack(features, degrees_one_hot))

            self.assigned_dimensions.append(self.features[-1].shape[1])
    
    def __getitem__(self, index):
        
        adjancency = self.adjacencies[index]
        
        # pad adjacency
        paddded_a = np.zeros((self.max_nodes, self.max_nodes))
        paddded_a[:self.num_nodes[index], :self.num_nodes] = adjancency

        return {
            'adjancency' : paddded_a, 
            'features' : self.features[index].copy(),
            'num_node' : self.num_nodes[index].copy(),
            'label' : self.labels[index].copy()                   
        }    
    
    def __len__(self,):
        return len(self.labels)


def load_data(directory, data_name):
    '''
    Returns a list of networkx graphs.
    '''
    
    # pathanmes

    edges_pathname = os.path.join(script_dir, "{}/{}_A.txt".format(directory, data_name))
    node_graph_pathname = os.path.join(script_dir, "{}/{}_graph_indicator.txt".format(directory, data_name))
    node_labels_pathname = os.path.join(script_dir, "{}/{}_node_labels.txt".format(directory, data_name))
    node_attributes_pathname =os.path.join(script_dir,  "{}/{}_node_attributes.txt".format(directory, data_name))
    graph_labels_pathname =os.path.join(script_dir,  "{}/{}_graph_labels.txt".format(directory, data_name))

    # mappings

    graph_edges = defaultdict(list)
    node_graph = defaultdict(int)

    # node to graph mapping 

    node_index = 0
    with open(node_graph_pathname, "r") as file:
        for line in file.readlines():
            node_graph[node_index] = int(line)
            node_index += 1


    with open(edges_pathname, "r") as file:
        # node_id = 0
        for line in file.readlines():
            node1, node2 = line.split(',')
            node1, node2 = int(node1), int(node2)
            if node_graph[node1] != node_graph[node2]:
                continue 
            graph_edges[node_graph[node1]] += [[node1, node2]]
            # graph_edges[node_id] += [[node1, node2]]
            
    # load the graph labels
    graph_labels = []
    with open(graph_labels_pathname) as file:
        lines = file.readlines()
        for line in lines:
            graph_labels.append(int(line) - 1)

    num_labels = max(graph_labels) + 1
    
    _, mapping = one_hot_encode_labels(list(set(graph_labels)))
    graph_labels = {graph_index : mapping[label] for graph_index, label in enumerate(graph_labels)}
    
    node_labels = load_node_information(node_labels_pathname, "labels")
    node_attributes = load_node_information(node_attributes_pathname, "node_attributes")

    graphs = []

    for graph_index, edges in graph_edges.items():
        Graph = nx.from_edgelist(edges)
        relabeling_mapping = {node : index for index, node in enumerate(Graph.nodes())}
        graph_attributes = {node : feature for node, feature in node_attributes.items() if node in Graph.nodes()}
        node_labels = {node : label for node, label in node_labels.items() if node in Graph.nodes()}
        nx.set_node_attributes(Graph, graph_attributes, "feature")
        nx.set_node_attributes(Graph, node_labels, "label")
        Graph = nx.relabel_nodes(Graph, relabeling_mapping)
        graphs.append(Graph)
        
    return graphs

def load_node_information(pathname, feature_type):
    node_features = []
    with open(pathname, "r") as file:
       lines = file.readlines()
       for line in lines:
           line = line.strip("\n")
           appending = int(line) if feature_type == "label" else np.array([float(attr) for attr in line.split(',')])
           node_features.append(appending)

    if feature_type == "label":
        _, encoding = one_hot_encode_labels(node_features)
        return {node : encoding[feature] for node, feature in enumerate(node_features)}
    return {node : feature for node, feature in enumerate(node_features)}




def cross_validation(graphs, test_index, max_nodes, arguments):

    '''
    returns a corss-validation dataset (torch dataloader) [training and testing folds]
    '''

    max_deg = None

    fold_size = len(graphs) // 10
    train_graphs = graphs[:test_index*fold_size]
    test_graphs = graphs[test_index * fold_size : (test_index + 1) * fold_size]
    
    if test_index < 9:
        train_graphs += graphs[(test_index + 1) * fold_size : ]

    train_dataset = Dataset(train_graphs, test_index, max_deg, 10)
    test_dataset = Dataset(test_graphs, 'structure', max_deg, max_nodes)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=arguments.batchsize, num_workers=arguments.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=arguments.batchsize, num_workers=arguments.num_workers)


    return train_dataloader, test_dataloader