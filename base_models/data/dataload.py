import numpy as np
import scipy.sparse
import torch
import scipy

def one_hot_encode_labels(labels):
    ''' one hot encoding of labels '''
    label_set = set(labels)
    mapping = {label : np.identity(len(label_set))[:,index] for index, label in enumerate(label_set)}
    labes_encoded = list(map(mapping.get, labels))

    return labes_encoded


def normalize(matrix):

    rowsum = np.sum(matrix, 1)
    inverses = np.power(rowsum, -1)
    inverses[np.isinf(inverses)] = 0.

    degree_matrix = torch.FloatTensor(scipy.sparse.diags(inverses).todense())
    return torch.matmul(degree_matrix, matrix)


def validation(output, labels):

    '''Predictions is a tensor and tensor.max returns max and index (predicted class)'''

    predictions = output.max(1)[1]
    correct = predictions.eq(labels).sum()
    
    return correct / len(labels)
    

def load_data(directory, data_name):
    print('Loading {} data ... '.format(data_name))

    # load data
    dataset = np.genfromtxt("{}/{}.content".format(directory, data_name))
    features = torch.FloatTensor(dataset[:, 1:-1])
    labels = torch.FloatTensor(one_hot_encode_labels(dataset[:, -1]))

     # create graph
    edges = np.genfromtxt("{}/{}.cites".format(directory, data_name))
    nodes = dataset[:, 0]
    node_index_map = {node : index for index, node in enumerate(nodes)}
    edges = np.array(list(map(node_index_map.get, edges.flatten())), dtype=np.float32).reshape(edges.shape)
    directed_adjacency = scipy.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[0:1])), shape=(len(nodes), len(nodes)), dtype=np.float32)

    adjcency_matrix = torch.FloatTensor(directed_adjacency.transpose() + directed_adjacency)

    train_indices = torch.LongTensor(range(1000))
    validate_indices = torch.LongTensor(range(1001, 2000))
    testing_indices = torch.LongTensor(range(2001, 2708))

    return normalize(adjcency_matrix), normalize(features), labels, train_indices, validate_indices, testing_indices
    

load_data('./dataset', 'cora')