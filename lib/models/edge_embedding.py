import torch
import torch.nn as nn


class EdgeEmbedding(nn.Module):
    def __init__(self,
                 num_features=3,
                 output_dim=64,
                 hidden_dim=64,
                 num_layer=2):
        super(EdgeEmbedding, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        # linear transforms
        for i in range(num_layer-1):
            self.add_module('W_{}'.format(
                i), nn.Linear(num_features, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, edge_attr: torch.Tensor):
        for i in range(self.num_layer-1):
            features = torch.relu(getattr(self, 'W_{}'.format(i))(edge_attr))
        features = self.final_layer(features)
        return features
