import numpy as np
import scipy.sparse
import torch
import scipy

def one_hot_encode_labels(labels):
    ''' one hot encoding of labels '''
    label_set = set(labels)
    mapping = {label : np.identity(len(label_set))[:,index] for index, label in enumerate(label_set)}
    labes_encoded = list(map(mapping.get, labels))

    return labes_encoded, mapping


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