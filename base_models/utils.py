import argparse
import torch

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning Rate")
    parser.add_argument('--hidden_dimension', type=int, default=16, help="dimension of the hidden features")
    parser.add_argument('--number_of_classes', type=int, default=7, help='number of classes')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,help='Number of epochs to train.')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
    parser.add_argument('--fastmode', default=False, help="evaluate while training")
    parser.add_argument("--n_embedding_layers", default=2, help="number of convolutional layers in the embdding bloick")
    parser.add_argument('--n_prediction_layers', default=2, help="number of convolutional layers in the prediction bloick")
    parser.add_argument('--pred_hidden_dimension', default=16, help="dimension for hte hidden features in prediction layers")
    parser.add_argument('--embedding_dimension', default=16, help="dmiension of the embdding")

    return parser



def embeddings_from_file(filename, n_nodes, dimension):
    embeddings = torch.FloatTensor(n_nodes, dimension)
    with open(filename, "r") as file:
        lines = file.readlines()
        for index, embedding_ in lines:
            embedding = torch.Tensor([float(num) for num in embedding_])
            embeddings[index, :] = embedding
        
    return embeddings