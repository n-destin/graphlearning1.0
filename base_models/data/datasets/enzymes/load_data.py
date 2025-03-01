import torch
import numpy as np
from collections import defaultdict
from datasets.data_utils import one_hot_encode_labels
import networkx as nx 

def load_data(directory, data_name):
    '''
    Get edges form
    '''
    
    # pathanmes

    edges_pathname = "{}/{}_A.txt".format(directory, data_name)
    node_graph_pathname = "{}/{}_graph_indicator.txt".format(directory, data_name)
    node_labels_pathname = "{}/{}_node_labels.txt".format(directory, data_name)
    node_attributes_pathname = "{}/{}_node_attributes.txt".format(directory, data_name)

    # mappings

    graph_edges = defaultdict(list)
    node_graph = defaultdict(int)

    # node to graph mapping 
    node_index = 0
    with open(node_graph_pathname, "r") as file:
        node_graph[int(line)] = node_index
        node_index += 1


    with open(edges_pathname, "r") as file:
        node_id = 0
        for line in file.readlines():
            node1, node2 = line.split(',')
            node1, node2 = int(node1), int(node2)
            if node_graph[node1] != node_graph[node2]:
                continue 
            graph_edges[node_graph[node1]] += [[node1, node2]]
            graph_edges[node_id] += [[node1, node2]]
    
    # load the graph labels
    graph_labels = list()
    graph_labels_pathname = "{}/{}_graph_labels".format(directory, data_name)
    with open(graph_labels_pathname) as file:
        lines = file.readlines()
        for line in lines:
            graph_labels.append(int(line))

    labels_one_hot_encoding = one_hot_encode_labels(list(set(graph_labels)))
    graph_labels = {graph : labels_one_hot_encoding[label] for graph, label in enumerate(graph_labels)}
    node_labels = load_node_information(node_labels_pathname)
    node_attributes = load_node_information(node_attributes_pathname)
    

    graphs = []

    for graph_index, edges in graph_edges.items():
        Graph = nx.from_edgelist(edges)
        graphs.append(Graph)
    
    return graphs

    # create graphs



def load_node_information(pathname):
    node_labels = []
    one_hot_encode = False
    with open(pathname, "r") as file:
       lines = file.readlines()
       for feature in enumerate(lines):
           one_hot_encode  = len(feature) == 1
           node_labels.append(feature)
    
    if one_hot_encode:
        encoding = one_hot_encode_labels(node_labels)
        return {node : encoding[feature] for node, feature in enumerate(node_labels)}
    return {node : feature for node, feature in enumerate(node_labels)}

        
load_data(".", "enzymes")