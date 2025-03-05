PyTorch Implementation of Graph Convolutional Networks based on the papers:

- https://arxiv.org/pdf/1606.09375
- https://arxiv.org/pdf/0912.3848
- https://arxiv.org/pdf/1609.02907.pdf

![alt text](graph_convolutional_networks.png)

Note: The original paper implements GCN using ReLU as the activation function. This repository uses softmax. 


Output Embeddings File:

Contains Embeddings of 100 nodes
Content on line i in the cora_gcn_embeddings.txt correcpond to the embedding of the node with id i. values have only four floating points.