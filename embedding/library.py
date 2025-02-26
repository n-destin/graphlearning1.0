from collections import defaultdict
import random
from embedding.testing_graphs import graph_test_one
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
    def __init__(self, value, embedding):
        self.value = value
        self.embedding = embedding
        self.parent = None
        self.right = None
        self.left  = None
        self.type = None

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

        current = set()
        for edge in self.edges:
            if len(edge) == 2:
                node1, node2 = edge
                weight = 1
            else:
                node1, node2, weight = edge
            self.mapping[node1][node2] = weight
            self.mapping[node2][node1] = weight
            self.node_embeddings[node1] = torch.rand(self.embdding_dimension, requires_grad=True)
            self.node_embeddings[node2] = torch.rand(self.embdding_dimension, requires_grad=True)
            treenode1 = HuffmanTreeNode(node1, self.node_embeddings[node1])
            treenode2 = HuffmanTreeNode(node2, self.node_embeddings[node2])
            self.node_treenode[node1] = treenode1
            self.node_treenode[node2] = treenode2
            current.add(treenode1)
            current.add(treenode2)
        current = list(current)
        
        while len(current) > 1:
            if len(current) % 2 == 1:
                current.append(HuffmanTreeNode(None))
            index = 0
            new_current = list()
            while index < len(current):
                node1 = current[index]
                node2 = current[index + 1]
                treenode = HuffmanTreeNode(str(index) + str(len(current)))
                treenode.embedding = torch.randn(self.embdding_dimension, requires_grad=True)
                treenode.right = node1
                treenode.left = node2
                node1.parent = treenode
                node1.type = "right"
                node2.parent = treenode
                node2.type  = "left"
                index += 2
                new_current.append(treenode)
            current = new_current

        self.HuffmanTree = current[0]

    def hierarchical_softamx(self, node, context):
        node = self.node_treenode(node) 
        returning = None
        while node:
            logit = torch.matmul(node.embedding, self.node_embeddings[context])
            if node.type == "right":
                probability = torch.softmax(logit)
            else:
                probability = torch.softmax(1 - logit)
        return probability
        
        
    def traverse(self, node, count, choosing, previous = None):
            if count == 0:
                return []
            previous = self.traverse(choosing(list(self.mapping[node].keys()), node, previous), count - 1, choosing, node)
            previous = [node] + previous
            return previous

    def random_choosing(self, nodes, current = None, previous = None):
        return random.choice(nodes)  

    def node_to_vec_random_choosing(self, nodes, current, previous):
        distribution = defaultdict(int)
        normalizer = 0
        for node in nodes:
            weight = self.mapping[current][node]
            if previous and node == previous:
                distribution[node] = weight
            elif previous and node in self.mapping[previous].keys():
                distribution[node] = 1 / self.return_parameter * weight
            else:
                distribution[node] = 1/self.in_out_parameter * weight
            normalizer += distribution[node]
        
        starting = 0
        range_node = defaultdict(int)
        
        for node, value in distribution.items():
            distribution[node]/= normalizer
            range_node[(starting, starting + distribution[node])] = node
            starting += distribution[node] 

        choosing = random.randint(0, 1)
        for range, node in range_node.items():
            start, end = range
            if (choosing >= start and choosing <= end )or abs(choosing - start) <= 0.1 or abs(choosing - end) <= 0.1: 
                return node
        return  None

        
    def random_walk(self, walk_length, type = None):

        returning = defaultdict(int) 
        if type == "deepwalk":
            choosing_function = self.random_choosing
        else:
            choosing_function = self.node_to_vec_random_choosing 
        for node in list(self.mapping.keys()):
            returning[node] = self.traverse(node, walk_length, choosing_function, None)
        return returning

    def skip_gram(self, walk, context_window, learning_rate, softmax_type = "logisitic"):
        normalizig_term = 0
        for node in self.node_embeddings.keys():
            for node_ in self.node_embeddings.keys():
                normalizig_term += torch.matmul(self.node_embeddings[node], self.node_embeddings[node_])
        if context_window >= len(walk):
            raise Exception("Reduce your context window")
        for index in range(len(walk)):
            left_index = max(0, index - context_window)
            right_index = min(len(walk), index + context_window + 1)
            for context in walk[left_index : right_index]:
                probability = torch.log(torch.matmul(self.node_embeddings[context], self.node_embeddings[walk[index]])) / normalizig_term
                if softmax_type == "hierarchical_softmax":
                    probability = self.hierarchical_softamx(walk[index], context)
                probability.backward(retain_graph=True)
                new_embedding = self.node_embeddings[walk[index]] - learning_rate * self.node_embeddings[walk[index]].grad
                self.node_embeddings[walk[index]] = torch.tensor(new_embedding, requires_grad=True)
