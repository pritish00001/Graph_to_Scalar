import torch
from initialisation import batch_build_topk_adjacency, batch_compute_eigen, batch_compute_normalized_laplacian, refine_all_X, train_model
from load_data import load_train_data_as_list
from modular import GraphEigenDataset
from preprocess import preprocess_graph_list_inplace
from torch_geometric.loader import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)

# Load and preprocess training data
training_graphs = load_train_data_as_list()
scaler = preprocess_graph_list_inplace(training_graphs, strategy='mean', device=device)
print("Preprocessing complete.")

# Create dataset objects
K1 = 10
K2 = 10

train_dataset = GraphEigenDataset(training_graphs,K1,K2)

# Create DataLoaders
BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("DataLoader created.")

# Initialization process
init_model = train_model(train_loader, input_dim=training_graphs[0].num_node_features, hidden_dim=64, output_dim=1, lr=0.01, epochs=2)
refined_X, _ = refine_all_X(init_model, train_loader, n_syn= 20, epochs=2 ,device=device)
print("refined_X.shape:", refined_X.shape)    
print("Refinement complete.")

refined_adj = batch_build_topk_adjacency(refined_X, Z=5)
print("A.shape:", refined_adj.shape)    
refined_L = batch_compute_normalized_laplacian(refined_adj)
refined_U, _ = batch_compute_eigen(refined_L, K1, K2)
print("Eigen decomposition complete.")

print("Refined X shape:", refined_X.shape)
print("Refined U shape:", refined_U.shape)
