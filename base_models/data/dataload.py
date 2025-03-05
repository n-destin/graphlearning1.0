import torch
import scipy
import scipy.sparse as sp
import numpy as np
from .datasets.data_utils import one_hot_encode_labels, normalize


def load_data(directory, data_name):
    print('Loading {} data ... '.format(data_name))

    # load data
    dataset = np.genfromtxt("{}/{}.content".format(directory, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(dataset[:, 1:-1], dtype=np.float32)

    labels = dataset[:, -1]
    _, labels_encoding = one_hot_encode_labels(list(set(dataset[:, -1])))
    labels_to_return = np.stack([np.array(labels_encoding[label]) for label in labels])
  
     # create graph
    edges = np.genfromtxt("{}/{}.cites".format(directory, data_name))
    nodes = dataset[:, 0].astype(float)

    node_index_map = {node : index for index, node in enumerate(nodes)}

    edges_shape = edges.shape
    edges = np.array(list(map(node_index_map.get, edges.flatten())), dtype=np.float32).reshape(edges_shape)

    print(edges.shape)

    directed_adjacency = scipy.sparse.coo_matrix((np.ones(edges_shape[0]), (edges[:, 0], edges[ :, 1])), shape=(len(nodes), len(nodes)), dtype=np.float32)

    adjcency_matrix = torch.FloatTensor(directed_adjacency.todense().transpose() + directed_adjacency.todense())

    train_indices = torch.LongTensor(range(1000))
    validate_indices = torch.LongTensor(range(1001, 2000))
    testing_indices = torch.LongTensor(range(2001, 2708))

    print("""
               Data loading finished
          ------ Dataset Statisitcis ---- 

          Shapre of returned tensors: 
          
          Features : {}
          Training_indices : {}
          Validation_indices : {}
          testing_indices : {}
          labels dimensions : {}

        """.format(features.shape, train_indices.shape, validate_indices.shape, testing_indices.shape, labels.shape))

    return normalize(adjcency_matrix), normalize(features.todense()), torch.Tensor(labels_to_return), train_indices, validate_indices, testing_indices
    