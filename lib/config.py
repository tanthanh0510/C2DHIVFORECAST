import torch

from torch.nn import Linear

from lib.models.edge_embedding import EdgeEmbedding
from lib.models.node_embedding import NodeEmbedding
from lib.models.model import Model

from lib.datasets.dataset import Datasets


class Config:
    def __init__(self, args, device):
        self.config = {}
        self.load_config(args)
        self.device = device

    def load_config(self, args):
        self.config["datasets"] = {}
        self.config["datasets"]["rootDir"] = args.root
        self.config["datasets"]["datasetName"] = args.dataset
        self.config["datasets"]["timeOffset"] = args.offset
        self.config["datasets"]["scale"] = args.scale
        self.config["datasets"]["offsetType"] = args.offsetType
        self.config["datasets"]["predictOffset"] = args.predictOffset

        self.config["model"] = {}
        self.config["model"]["node_embedding_dim"] = args.node_dim
        self.config["model"]["edge_embedding_dim"] = args.edge_dim
        self.config["model"]["pooling"] = args.pooling
        self.config["model"]["graph_block"] = args.graph_block
        self.config["model"]["pooling_params"] = args.pooling_params
        self.config["model"]["graph_block_params"] = args.graph_block_params

        self.config["optimizer"] = {}
        self.config["optimizer"]["name"] = "Adam"
        self.config["optimizer"]["parameters"] = {}
        self.config["optimizer"]["parameters"]["lr"] = 5e-4

        self.config["loss"] = 'MSELoss'

        self.config["optimizer"]["parameters"]["weight_decay"] = 1e-5

        self.config['lr_scheduler'] = {}
        self.config['lr_scheduler']['name'] = "CosineAnnealingWarmRestarts"
        self.config["lr_scheduler"]["parameters"] = {'T_0': 10}

        self.config["exp_name"] = args.exp_name

    def get_dataset(self, **kwargs):
        self.dataset = Datasets(**kwargs, **self.config["datasets"])
        return self.dataset

    def get_model(self, **kwargs):
        node_embedding = NodeEmbedding(
            len(self.dataset.infectiousObject),
            len(self.dataset.occupation),
            len(self.dataset.infectionPathway),
            len(self.dataset.sexs),
            len(self.dataset.ageGroups),
            node_embedding_dim=self.config['model']['node_embedding_dim']
        )
        edge_embedding = EdgeEmbedding(
            3,
            output_dim=self.config['model']['edge_embedding_dim']
        )
        self.config['model']['num_location'] = len(self.dataset.locationName)
        if self.config['model']['graph_block'] != 'gin':
            self.config['model']['graph_block_params']['edge_dim'] = self.config['model']['edge_embedding_dim']
            self.config['model']['graph_block_params']['in_channels'] = self.config['model']['node_embedding_dim']
        else:
            self.config['model']['graph_block_params']['nn'] = Linear(
                self.config['model']['node_embedding_dim'], self.config['model']['node_embedding_dim']
            )

        model = Model(node_embedding, edge_embedding,
                      **self.config['model'], **kwargs)

        return model

    def get_optimizer(self, model_parameters):
        return getattr(torch.optim, self.config['optimizer']['name'])(model_parameters,
                                                                      **self.config['optimizer']['parameters'])

    def get_loss_function(self, **kwargs):
        return getattr(torch.nn, self.config['loss'])(**kwargs)

    def get_lr_scheduler(self, optimizer):
        return getattr(torch.optim.lr_scheduler,
                       self.config['lr_scheduler']['name'])(optimizer, **self.config['lr_scheduler']['parameters'])

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
