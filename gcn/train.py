import torch
from torch import optim
import argparse
from model import ConvolutionalGraphNetwork
from data.dataload import load_data, validation

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning Rate")
parser.add_argument('--hidden_dimension', type=int, default=16, help="dimension of the hidden features")
parser.add_argument('--number_of_classes', type=int, default=7, help='number of classes')
parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--fastmode', default=False, help="evaluate while training")

arguments = parser.parse_args()
arguments.cuda = not arguments.no_cuda and torch.cuda.is_available()

normalized_adjacency, features, labels, train_indices, validate_indices, test_indices = load_data('.data/dataset', 'cora')
model = ConvolutionalGraphNetwork(labels.shape[1], arguments.hidden_dimension, arguments.number_of_classes, 2)



if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
    normalized_adjacency = normalized_adjacency.cuda()
    features = features.cuda()
    labels = labels.cuda()
    model = model.cuda()

optimizer = optim.Adam(model.parameters, lr=arguments.learning_rate, weight_decay=arguments.weight_decay)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    predictions = model(features[train_indices], normalized_adjacency)
    loss = torch.nn.functional.nll_loss(predictions[train_indices], labels[train_indices])
    train_accuracy = validation(predictions[train_indices], labels[train_indices])
    loss.backward()
    optimizer.step()

    if arguments.fastmode:
        model.eval()
        predictions = model(features, normalized_adjacency)
    loss = torch.nn.functional(predictions[validate_indices], normalized_adjacency)
    validation_accuracy = validation(predictions[validate_indices], labels[validate_indices])

    print('Epoch {}, Validation accuracy {} Training accuracy {} TrainingLoss {}'.format(epoch, validation_accuracy, train_accuracy, loss.item()))


def test():
    model.eval()
    output = model(features, normalized_adjacency)
    loss = torch.nn.functional.nll_loss(output[validate_indices], labels[validate_indices])
    accuracy = validation(output[validate_indices], labels=[validate_indices])

    print("Validation Loss: {}, Validation accuracy: {}".format(loss.item(), accuracy))


for epoch in arguments.epochs:
    train(epoch)

test()