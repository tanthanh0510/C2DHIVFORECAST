import torch
from torch import nn
from torch.nn import Linear

from torch_geometric.nn import (
    TransformerConv,
    GATConv,
    GATv2Conv,
    GINConv,
    SAGPooling,
    TopKPooling,
    GCNConv,
    global_mean_pool

)

mapPool = {
    "mean": global_mean_pool,
    "sag": SAGPooling,
    "topk": TopKPooling
}
#  GAT, GAT2, GIN, TRANSFORMER
mapGraphBlock = {
    "transformer": TransformerConv,
    "gat": GATConv,
    "gat2": GATv2Conv,
    "gin": GINConv
}


class Model(nn.Module):
    def __init__(self,
                 node_embedding: nn.Module,
                 edge_embedding: nn.Module,
                 node_embedding_dim: int = 64,
                 edge_embedding_dim: int = 64,
                 num_location: int = 24,
                 output_dim: int = 1,
                 pooling: str = "graph",
                 graph_block: str = "transformer",
                 **kwargs):
        super(Model, self).__init__()
        self.node_embedding = node_embedding
        self.edge_embedding = edge_embedding
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.pooling_name = pooling
        self.num_location = num_location
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=0.2)

        graph_block_params = kwargs.get("graph_block_params", {
            "in_channels": self.node_embedding_dim,
            "out_channels": self.output_dim
        })
        self.graph_block_name = graph_block
        self.graph_block = mapGraphBlock[graph_block](**graph_block_params)

        self.relu = nn.ReLU(inplace=True)
        output_dim_graph = graph_block_params.get(
            "out_channels", self.node_embedding.hidden_dim)

        if hasattr(self.graph_block, "concat") and hasattr(self.graph_block, "heads"):
            if self.graph_block.concat:
                output_dim_graph *= self.graph_block.heads

        if pooling != "mean":
            self.gat = GATConv(output_dim_graph, output_dim_graph)
            self.edge_local_embedding = Linear(1, self.edge_embedding_dim)
            pooling_params = kwargs.get("pooling_params", {})
            if pooling_params:
                pooling_params["in_channels"] = output_dim_graph
                self.pooling = mapPool[pooling](**pooling_params)
            else:
                self.pooling = mapPool[pooling]()
                print("Pooling used default parameters.")

        else:
            self.pooling = mapPool[pooling]
        self.fc1 = Linear(output_dim_graph, 2*output_dim_graph)
        self.fc2 = Linear(2*output_dim_graph, self.output_dim)

    def forward(self, data: dict, **kwargs):
        node_embedding = self.node_embedding(data.nodes, data.batch, **kwargs)
        edge_embedding = self.edge_embedding(data.edge_attr)
        x = self.dropout(node_embedding)
        skip_connection = x
        if self.graph_block_name == "gin":
            x = self.graph_block(x=x, edge_index=data.edge_index)
        else:
            x = self.graph_block(x=x, edge_index=data.edge_index,
                                 edge_attr=edge_embedding)

        x += skip_connection
        x = self.relu(x)
        batch = data.nodes["location"] + self.num_location*data.batch
        if self.pooling_name != "mean":
            num_edge_local = data.nodes['num_edge_local']
            num_node_local = data.ptr
            edge_index_local = data.nodes['location_edge']
            edge_attr_local = data.nodes['location_edge_attr']
            edge_attr_local = self.edge_local_embedding(edge_attr_local)
            i = 0
            for j in range(len(num_edge_local)):
                edge_index_local[i:num_edge_local[j]+i,
                                 :] = edge_index_local[i:num_edge_local[j]+i, :]+num_node_local[j]
                i = num_edge_local[j]+i
            edge_index_local = edge_index_local.transpose(1, 0)
            edge_index_local = edge_index_local.to(x.device)
            x, edge_index, edge_attr, batch, _, _ = self.pooling(x, edge_index_local, edge_attr_local,
                                                                 batch=batch)
            x = self.gat(x, edge_index, edge_attr)
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
