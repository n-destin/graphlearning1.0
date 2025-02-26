from .library import Graph
from testing_graphs import graph_test_one

def deepwalk(graph, walks_per_node, walk_length, embedding_dimension, window_size):
    graph = Graph(graph_test_one, embedding_dimension)
    for _ in walks_per_node:
        random_walks = graph.randomwalk(walk_length, "deepwalk")
        for node, walk in random_walks.items():
            graph.skipgram(walk, window_size)
    
    return graph.node_embeddings
