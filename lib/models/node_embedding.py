import torch
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm


class NodeEmbedding(nn.Module):
    def __init__(self,
                 num_infectious_object,
                 num_occupation,
                 num_infection_route,
                 num_sex,
                 num_age_grp,
                 node_embedding_dim=64):
        super(NodeEmbedding, self).__init__()
        self.num_infectious_object = num_infectious_object
        self.num_occupation = num_occupation
        self.num_infection_route = num_infection_route
        self.num_sex = num_sex
        self.hidden_dim = node_embedding_dim
        self.infectious_object_embedding = nn.Embedding(
            num_infectious_object, self.hidden_dim)
        self.occupation_embedding = nn.Embedding(
            num_occupation, self.hidden_dim)
        self.infection_route_embedding = nn.Embedding(
            num_infection_route, self.hidden_dim)
        self.sex_embedding = nn.Embedding(num_sex, self.hidden_dim)
        self.phys_pos_embedding = nn.Linear(2, self.hidden_dim)
        self.times_embedding = nn.Linear(1, self.hidden_dim)
        self.new_case_embedding = nn.Linear(1, self.hidden_dim)
        self.age_grp_embedding = nn.Embedding(num_age_grp, self.hidden_dim)
        self.norm = GraphNorm(node_embedding_dim)

    def forward(self, node: dict, batch: torch.Tensor, **kwargs):
        feature_ignore = kwargs.get('feature_ignore', None)
        new_case = self.new_case_embedding(node['new_case'].view(-1, 1))
        times = self.times_embedding(node['time'].view(-1, 1))
        if feature_ignore != 'all':
            if feature_ignore == 'infectious_object':
                infectious_object = torch.zeros(
                    node['infectious_object'].shape[0], self.hidden_dim).to(
                        node['infectious_object'].device)
            else:
                infectious_object = self.infectious_object_embedding(
                    node['infectious_object'])
            if feature_ignore == 'occupation':
                occupation = torch.zeros(
                    node['occupation'].shape[0], self.hidden_dim).to(
                        node['occupation'].device)
            else:
                occupation = self.occupation_embedding(node['occupation'])
            if feature_ignore == 'infection_route':
                infection_route = torch.zeros(
                    node['infection_route'].shape[0], self.hidden_dim).to(
                        node['infection_route'].device)
            else:
                infection_route = self.infection_route_embedding(
                    node['infection_route'])
            if feature_ignore == 'sex':
                sex_embedding = torch.zeros(
                    node['sex'].shape[0], self.hidden_dim).to(
                        node['sex'].device)
            else:
                sex_embedding = self.sex_embedding(node['sex'])
            if feature_ignore == 'phys_pos':
                phys_pos = torch.zeros(
                    node['phys_pos'].shape[0], self.hidden_dim).to(
                        node['phys_pos'].device)
            else:
                phys_pos = self.phys_pos_embedding(node['phys_pos'])
            if feature_ignore == 'age_grp':
                age_grp = torch.zeros(
                    node['age_grp'].shape[0], self.hidden_dim).to(
                        node['age_grp'].device)
            else:
                age_grp = self.age_grp_embedding(node['age_grp'])
            
            x = infectious_object + occupation + infection_route + \
                sex_embedding + phys_pos + times + new_case + age_grp
        else:
            x = new_case + times
        x = self.norm(x, batch)
        return x
