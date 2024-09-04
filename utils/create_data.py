from torch_geometric.data import Batch


def collate(dataset):
    batch_data = Batch.from_data_list([data for data in dataset])
    return batch_data
