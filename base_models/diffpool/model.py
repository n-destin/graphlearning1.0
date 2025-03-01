import torch
from torch import nn
from base_models.graphsage.model import GraphSage


class DiffPoll(nn.Module):
    def __init__(self, graph_sage_depth, graph_sage_pooling_type, graph_sage_dimension):
        super().__init__()
        self.pooling_model = GraphSage()
        
