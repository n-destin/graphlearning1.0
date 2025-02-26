from collections import defaultdict
import random
from testing_graphs import graph_test_one
import math
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch import optim
import numpy as np


def sigmoid(number):
    return 1/(1+math.exp(number))

def softmax(list):
    return [element / sum(list) for element in list]\

class HuffmanTreeNode():
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.right = None
        self.left  = None

class Graph():
    def __init__(self, edges, embedding_dimension):
        self.nodes = []
        self.embdding_dimension = embedding_dimension
        self.edges = edges
        self.node_embeddings = defaultdict(int)
        self.mapping = defaultdict(lambda: defaultdict(int))
        self.return_parameter = None
        self.in_out_parameter = None
        self.HuffmanTree = None
        self.node_treenode = defaultdict()
        
        self.create_graph()

    def test_graph(self, ):
        pass
    
    def create_graph(self,):

        current = list()
        for edge in self.edges:
            if len(edge) == 2:
                node1, node2 = edge
                weight = 1
            else:
                node1, node2, weight = edge
            self.mapping[node1][node2] = weight
            self.mapping[node2][node1] = weight
            self.node_embeddings = torch.rand(self.embdding_dimension, requires_grad=True)
            self.node_embeddings.add(node1)
            self.nodes.add(node2)
            treenode1 = HuffmanTreeNode(node1)
            treenode2 = HuffmanTreeNode(node2)
            self.node_treenode[node1] = treenode1
            self.node_treenode[node2] = treenode2
            current.append(treenode1)
            current.append(treenode2)

         # construct the huffman tree for Hierarchical Softmax
        if len(current) % 2 == 1:
            current.append(HuffmanTreeNode(None))

        while len(current) > 0:
            index = 0
            while index < len(current):
                node1 = current[index]
                node2 = current[index + 1]
                treenode = HuffmanTreeNode(str(index) + str(len(current)))
                treenode.right = node1
                treenode.left = node2
                node1.parent = treenode
                node2.parent = treenode
                index += 2

        self.HuffmanTree = current[0]
    
    def traverse(self, node, count, choosing):
            if count == 0:
                return []
            previous = self.traverse(choosing(self.mapping[node]), count - 1)
            previous.append(node)
            return previous

    def random_choosing(self, nodes):
        return random.choose(nodes)  

    def node_to_vec_random_choosing(self, nodes, current, previous):
        current = None
        pi_previous_next = 0
        compare = pi_previous_next
        for node in nodes:
            weight = self.mapping[current][node]
            if node  == previous and pi_previous_next < weight:
                pi_previous_next = weight
            elif node in self.mapping[previous].keys() and pi_previous_next < 1/self.return_parameter * weight:
                pi_previous_next = 1/self.return_parameter * weight
            elif 1/self.in_out_parameter * weight > pi_previous_next:
                pi_previous_next = 1/self.in_out_parameter * weight
        if compare < pi_previous_next:
            current = node
        return current

        
    def random_walk(self, walk_length, type = None):
        returning = defaultdict(int) 
        if type == "deepwalk":
            choosing_function = self.random_choosing
        else:
            choosing_function = self.node_to_vec_random_choosing
        for node in self.nodes:
            returning[node] = self.traverse(node, walk_length, choosing_function)
        return returning

    def skip_gram(self, walk, context_window, learning_rate, softmax = True):
        normalizig_term = 0
        for node in self.node_embeddings.keys():
            for node_ in self.node_embeddings.keys():
                normalizig_term += torch.matmul(self.node_embeddings[node], self.node_embeddings[node_])
        if context_window >= walk:
            raise Exception("Reduce your context window")
        for index in range(len(walk)):
            left_index = max(0, index - context_window)
            right_index = min(len(walk), index + context_window + 1)
            for context in walk[left_index : right_index]:
                probability = math.log(torch.matmul(self.node_embeddings[context], self.node_embeddings[walk[index]])) / normalizig_term
                probability.backwards()
                self.node_embeddings[node] -= learning_rate * self.node_embeddings.grad()
        