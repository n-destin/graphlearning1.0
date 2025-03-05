import numpy as np
import scipy.sparse
import torch
import scipy

def one_hot_encode_labels(labels):
    ''' one hot encoding of labels '''
    label_set = set(labels)
    mapping = {label : np.identity(len(label_set))[:, index] for index, label in enumerate(label_set)}
    labes_encoded = list(map(mapping.get, labels))

    return labes_encoded, mapping


def normalize(matrix):
    print(matrix.shape)
    matrix = np.array(matrix)
    rowsum = np.sum(matrix, axis=1)
    inverses = np.power(rowsum, -1)
    inverses[np.isinf(inverses)] = 0.

    degree_matrix = torch.FloatTensor(scipy.sparse.diags(inverses).todense())
    
    return torch.matmul(degree_matrix, torch.Tensor(matrix))


def validation(output, labels):
    '''Predictions is a tensor and tensor.max returns max and index (predicted class)'''
    predictions = output.max(1)[1]
    correct = predictions.eq(labels.max(1)[1]).sum()
    return correct / len(labels)


def write_summary(filename, arguments):

    summary = """
            ================= Result ============
            Epochs :        {}
            Dataset:        {}
            ------------ Best epoch -----------
            Train:          {}
            Test:           {}
            Best epoch:     {}
            ------------ Last epoch ------------
            Train:          {}
            Test:           {}
            ----------- Validation -------------
            Validation :    {}
            -------------------------------

            Arguments:
            {}
             """.format(*arguments)
    with open(filename, 'w') as file:
        file.write(summary)