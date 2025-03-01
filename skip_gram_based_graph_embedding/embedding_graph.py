from skip_gram_based_graph_embedding.library import Graph
from testing_graphs import graph_test_one

def graph_embedding(walks_per_node, walk_length, embedding_dimension, learning_rate, window_size, embedding_type):
    graph = Graph(graph_test_one, embedding_dimension)
    graph.in_out_parameter = 0.5
    graph.return_parameter = 0.5
    for _ in walks_per_node:
        random_walks = graph.random_walk(walk_length, embedding_type)
        for node, walk in random_walks.items():
            graph.skip_gram(walk, 2, learning_rate)
    with open("embddings_" + str(window_size) + "_" + walks_per_node + "_" + walk_length + "_" + embedding_type + ".txt", "w") as file:
            for node, embedding in graph.node_embeddings.items():
                file.write(str(node) + ": [") 
                for value in embedding:
                    file.write(str(value.item()))
                file.write(']\n')
