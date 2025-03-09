import torch
import os
from torch import nn
from base_models.data.datasets.enzymes.load_data import load_data
from base_models.diffpool.encoders import Encoder
from base_models.diffpool.encoders import MeanPooling
from base_models.data.datasets.enzymes.load_data import cross_validation
from base_models.utils import create_parser

script_dir = os.path.dirname(os.path.abspath(__file__))

parser = create_parser()
arguments = parser.parse_args()

graphs, node_labels, node_attributes, graph_labels = load_data('ENZYMES')
embeddings_pathname = os.path.join(script_dir, "../gcn/cora_gcn_embeddings.txt")

print(

    """
    ============== Dataset Summary=========
    number of graphs: {}, 
    total number of nodes : {}
    Node attributes dimension : {}, 
    nodes labels one hot encoded ? : {}
    """.format(len(graphs), len(node_labels), len(list(node_attributes.values())[0]), len(list(node_labels.values())[0]) > 1)
)

model = Encoder(None, arguments)
optimizer = torch.optim.Adam(model.parameters, lr=arguments.learning_rate, weight_decay=arguments.weight_decay)


if arguments.cuda:
    model = model.cuda()


# for test_index in range(10):
    

def train(epoch, test_index):
    training_graphs, testing_graphs = cross_validation(graphs, test_index, None, arguments)
    training_labels = torch.FloatTensor(len(training_graphs), arguments.output_dimension)
    testing_labels = torch.FloatTensor(len(testing_graphs), arguments.output_dimension)

    for index, graph in enumerate(training_graphs):
        training_labels[index][graph['label']] = 1

    for index, graph in enumerate(testing_graphs):
        testing_labels[index][graph['label']] = 1
    
    model.graphs = training_graphs
    model.train()
    predictions = model(training_graphs)
    loss = nn.functional.nll_loss(predictions, training_labels)

    loss.backward()
    
    if arguments.fastmode:
        model.eval()
        predictions = model(testing_graphs)
        loss = nn.functional.nll_loss(predictions, testing_labels)




for epoch in arguments.n_epochs:
    for test_index in range(10):
        train(epoch, test_index)    