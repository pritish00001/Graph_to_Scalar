import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)

def load_train_test():
    data_path = 'Example-1'
    train_loader = torch.load(os.path.join(data_path, 'data', 'train_dataset.pt'), weights_only=False, map_location='cpu')
    test_loader = torch.load(os.path.join(data_path, 'data', 'test_dataset_2.pt'), weights_only=False, map_location='cpu')
    return train_loader, test_loader

def load_train_data_as_list():
    data_path = 'Example-1'
    train_loader = torch.load(os.path.join(data_path, 'data', 'train_dataset.pt'), weights_only=False, map_location='cpu')

    individual_list_of_training_graphs = []
    for batch in train_loader:
        individual_graphs = batch.to_data_list()
        individual_graphs = [graph.to(device) for graph in individual_graphs]
        individual_list_of_training_graphs.extend(individual_graphs)

    for g in individual_list_of_training_graphs:
        g.y = g.stiffness
        del g.stiffness  # optional but cleaner

    return individual_list_of_training_graphs