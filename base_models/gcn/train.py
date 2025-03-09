import torch
from torch import optim
from base_models.gcn.model import ConvolutionalGraphNetwork
from base_models.data.dataload import load_data
from base_models.data.datasets.data_utils import validation, write_summary
import numpy as np
from base_models.utils import create_parser


parser = create_parser()
arguments = parser.parse_args()
arguments.cuda = not arguments.no_cuda and torch.cuda.is_available()

normalized_adjacency, features, labels, train_indices, validate_indices, test_indices = load_data('../data/datasets/cora', 'cora')

# print(features.shape, 'printing features', features.shape[1])

# print(labels.shape, 'shape of the labels')

model = ConvolutionalGraphNetwork(features.shape[1], arguments)
summary_pathname = "{}_{}_summary.txt".format("cora", "gcn")
embeddings_pathname = "{}_{}_embeddings.txt".format("cora", "gcn")

if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
    normalized_adjacency = normalized_adjacency.cuda()
    features = features.cuda()
    labels = labels.cuda()
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
best_epoch_num = 0
best_train = 0
best_test = 0
last_train = None
last_test = None
embeddings_ = torch.FloatTensor((len(normalized_adjacency), arguments.hidden_dimension))


# arguments string 

arguments_string = ""
for arg, value in vars(arguments).items():
    arguments_string += "{} : {} \n\t\t\t".format(arg, value)


def train(epoch):

    global best_train, best_test, best_epoch_num, summary_pathname, embeddings_

    model.train()
    optimizer.zero_grad()
    embeddings, support = model(features, normalized_adjacency)
    # print("going to predict", embeddings.shape)
    predictions = model.predict(embeddings, normalized_adjacency)
    
    loss = torch.nn.functional.nll_loss(predictions[train_indices], torch.LongTensor(np.where(labels[train_indices])[1]))
    train_accuracy = validation(predictions[train_indices], labels[train_indices])
    loss.backward()
    optimizer.step()

    if arguments.fastmode: # evaluate while training
        model.eval()
        predictions, embeddings = model(features, normalized_adjacency)
        embeddings_ = embeddings
    loss = torch.nn.functional.nll_loss(predictions[test_indices], torch.LongTensor(np.where(labels[test_indices])[1]))
    test_accuracy = validation(predictions[test_indices], labels[test_indices])
    
    if train_accuracy > best_train:
        best_train = train_accuracy
        best_test = test_accuracy
        best_epoch_num = epoch

    if epoch % 40 == 0:
        print('{}, {}'.format(test_accuracy, epoch))
        last_train = train_accuracy if epoch == arguments.epochs else None
        last_test = test_accuracy if epoch == epoch ==  arguments.epochs else None
        write_summary(summary_pathname, [epoch, "cora", best_train, best_test, best_epoch_num, last_train, last_test, None, arguments_string])


def test():

    global best_train, best_test, best_epoch_num, summary_pathname, embeddings_

    model.eval()
    embeddings, support = model(features, normalized_adjacency)
    predictions = model.predict(embeddings, normalized_adjacency)
    embeddings_ = embeddings
    loss = torch.nn.functional.nll_loss(predictions[validate_indices], torch.LongTensor(np.where(labels[validate_indices])[1]))
    accuracy = validation(predictions[validate_indices], labels[validate_indices])

    write_summary(summary_pathname, [arguments.epochs, "cora", best_train, best_test, best_epoch_num, last_train, last_test, accuracy, arguments_string])
    np.savetxt(embeddings_pathname, embeddings_[:100, :].detach().numpy(), fmt="%.4f")



for epoch in range(arguments.epochs + 1):
    train(epoch)

test()