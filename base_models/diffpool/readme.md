The previous models in this repo were generally considering a task of node prediction. If we extend the task to graph prediction, where given a series of graph the model should ouput, for example, a class associated to each graph, our task is to learn general graph embedding, instead of one embedding for the downstream tasks. To generate graph embeddings, common approaches include simply
summing up or averaging all the node embeddings in a final layer introducing a “virtual node”
that is connected to all the nodes in the graph or aggregating the node embeddings using a deep
learning architecture that operates over sets  However, all of these methods have the limitation
that they do not learn hierarchical representations (intrinsic structures that the graphs have.) (i.e., all the node embeddings are globally pooled
together in a single layer). 

The contribution of diffpool to this problem can be summed into the following except from the paper:

```Our proposed approach, DIFFPOOL, addresses the above challenges by learning a cluster assignment
matrix over the nodes using the output of a GNN model. The key intuition is that we stack L GNN
modules and learn to assign nodes to clusters at layer l in an end-to-end fashion, using embeddings
generated from a GNN at layer l − 1. Thus, we are using GNNs to both extract node embeddings that
are useful for graph classification, as well to extract node embeddings that are useful for hierarchical
pooling.
```

![alt text](diffpool.png)

paper: https://arxiv.org/pdf/1806.08804