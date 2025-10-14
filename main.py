import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torch_geometric.data import Data, DataLoader
from torch.serialization import add_safe_globals

# =======================
# Import your modules
# =======================
from initialisation import (
    batch_build_topk_adjacency,
    batch_compute_eigen,
    batch_compute_normalized_laplacian,
    create_dataloader_and_save_list,
    refine_all_X,
    train_model,
    GraphModel,
)
from load_data import load_train_data_as_list
from modular import GraphEigenDataset
from preprocess import preprocess_graph_list_inplace
from models import GNN
from coupled_training import coupled_training_dataloaders
from losses import Losses

# =======================
# Setup
# =======================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = "saved_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)

random.seed(42)
torch.manual_seed(42)

# =======================
# 1️⃣ Load / Preprocess training graphs
# =======================
preprocessed_graphs_path = os.path.join(SAVE_DIR, "preprocessed_training_graphs.pt")
scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")

add_safe_globals([Data])  # PyTorch 2.6+ safe global for Data

if os.path.exists(preprocessed_graphs_path) and os.path.exists(scaler_path):
    print("Loading preprocessed data...")
    training_graphs = torch.load(preprocessed_graphs_path, map_location='cpu', weights_only=False)
    scaler = joblib.load(scaler_path)
else:
    print("Preprocessing training data...")
    training_graphs = load_train_data_as_list()
    scaler = preprocess_graph_list_inplace(training_graphs, strategy='mean', device=device)
    torch.save(training_graphs, preprocessed_graphs_path)
    joblib.dump(scaler, scaler_path)
    print(f"Preprocessed data saved in '{SAVE_DIR}'")

print("Preprocessing complete.")

# =======================
# 2️⃣ Dataset and DataLoader
# =======================
K1, K2 = 10, 10
dataset_path = os.path.join(SAVE_DIR, "train_dataset.pt")

if os.path.exists(dataset_path):
    print("Loading saved train dataset...")
    train_dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
else:
    print("Creating train dataset...")
    train_dataset = GraphEigenDataset(training_graphs, K1, K2)
    torch.save(train_dataset, dataset_path)
    print(f"Train dataset saved at '{dataset_path}'")

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("DataLoader created.")

# =======================
# 3️⃣ Model initialization
# =======================
model_path = os.path.join(SAVE_DIR, "initialization_model.pt")
input_dim = training_graphs[0].num_node_features
hidden_dim = 64
output_dim = 1

if os.path.exists(model_path):
    print("Loading existing initialized model...")
    init_model = GraphModel(input_dim, hidden_dim, output_dim).to(device)
    init_model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Training initialization model...")
    init_model = train_model(
        train_loader,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        lr=0.01,
        epochs=2
    )
    torch.save(init_model.state_dict(), model_path)
    print(f"Initialization model saved at '{model_path}'")

# =======================
# 4️⃣ Refinement phase
# =======================
refined_X_path = os.path.join(SAVE_DIR, "refined_X.pt")
refined_adj_path = os.path.join(SAVE_DIR, "refined_adj.pt")
refined_U_path = os.path.join(SAVE_DIR, "refined_U.pt")

if os.path.exists(refined_X_path) and os.path.exists(refined_U_path):
    print("Loading refined tensors...")
    refined_X = torch.load(refined_X_path, map_location=device)
    refined_U = torch.load(refined_U_path, map_location=device)
else:
    print("Running refinement...")
    n_syn = K1 + K2
    refined_X, _ = refine_all_X(init_model, train_loader, n_syn=n_syn, epochs=2, device=device)
    print("Refinement complete.")

    print("Computing adjacency, Laplacian, and eigen decomposition...")
    refined_adj = batch_build_topk_adjacency(refined_X, Z=5)
    refined_L = batch_compute_normalized_laplacian(refined_adj)
    refined_U, _ = batch_compute_eigen(refined_L, K1, K2)
    print("Eigen decomposition complete.")

    torch.save(refined_X, refined_X_path)
    torch.save(refined_adj, refined_adj_path)
    torch.save(refined_U, refined_U_path)
    print("Refined data saved.")

print("Refined X shape:", refined_X.shape)
print("Refined U shape:", refined_U.shape)

# =======================
# 5️⃣ Create synthetic graphs
# =======================
synthetic_list_path = os.path.join(SAVE_DIR, "synthetic_data_list.pt")

if os.path.exists(synthetic_list_path):
    print("Loading saved synthetic graphs...")
    synthetic_graph_list = torch.load(synthetic_list_path, map_location='cpu', weights_only=False)
else:
    print("Creating synthetic graphs...")
    _, synthetic_graph_list = create_dataloader_and_save_list(
        refined_X,
        refined_U,
        batch_size=1,
        shuffle=False,
        save_dir=SAVE_DIR
    )
    torch.save(synthetic_graph_list, synthetic_list_path)
    print("Synthetic graphs created and saved.")

# =======================
# 6️⃣ Make synthetic tensors learnable
# =======================
for data in synthetic_graph_list:
    data.x = nn.Parameter(data.x.clone().detach().requires_grad_(True))
    data.u = nn.Parameter(data.u.clone().detach().requires_grad_(True))

x_optimizer = torch.optim.Adam([d.x for d in synthetic_graph_list], lr=1e-3)
u_optimizer = torch.optim.Adam([d.u for d in synthetic_graph_list], lr=1e-3)

# =======================
# 7️⃣ Coupled training with GNN
# =======================

GNN_model = GNN().to(device)

final_Le, final_Lo, final_Lr = coupled_training_dataloaders(
    GNN_model=GNN_model,
    train_dataset=train_dataset,
    synthetic_graph_list=synthetic_graph_list,
    x_optimizer=x_optimizer,
    u_optimizer=u_optimizer,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    tau1=1,
    epochs=2,
    K1=K1,
    K2=K2,
    batch_size=BATCH_SIZE
)

# =======================
# 8️⃣ Save final trained GNN
# =======================
final_model_path = os.path.join(SAVE_DIR, "dist_model.pt")
torch.save(GNN_model.state_dict(), final_model_path)
print(f"\n✅ Coupled training complete. Model saved at: {final_model_path}")
print(f"Final Avg Losses — Le={final_Le:.6f}, Lo={final_Lo:.6f}, Lr={final_Lr:.6f}")
