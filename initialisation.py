import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GraphModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super(GraphModel, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, batch):
        """
        batch: a PyG Batch object
        batch.x -> node features of all graphs concatenated [total_nodes, input_dim]
        batch.batch -> graph assignment vector [total_nodes], tells which graph each node belongs to
        """
        node_embeddings = self.mlp1(batch.x)  # [total_nodes, hidden_dim]
        
        # Aggregate node embeddings per graph (mean pooling)
        graph_embeddings = torch_scatter.scatter_mean(
            node_embeddings, batch.batch, dim=0
        )  # [num_graphs_in_batch, hidden_dim]
        
        y_hat = self.mlp2(graph_embeddings)  # [num_graphs_in_batch, output_dim]
        return y_hat.squeeze(-1)  # make shape [num_graphs_in_batch]


def train_model(dataloader, input_dim=5, hidden_dim=64, output_dim=1, lr=0.01, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GraphModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            y_hat = model(batch)                  # [batch_size]
            y_true = batch.y.view(-1).float()     # [batch_size]

            loss = criterion(y_hat, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
        
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")
    
    return model

def refine_all_X(model, dataloader, n_syn, epochs=100, lr=0.1, device=None):
    """
    Refines synthetic node features for graphs in a dataloader using a single GraphModel.

    Args:
        model: trained GraphModel (expects PyG Batch input)
        dataloader: DataLoader yielding batches of PyG Data objects
        n_syn: fixed number of synthetic nodes per graph
        epochs: number of refinement steps
        lr: learning rate for updating synthetic X
        device: torch device

    Returns:
        refined_X: tensor of shape [total_graphs, n_syn, input_dim]
        final_preds: tensor of shape [total_graphs]
    """
    criterion = nn.MSELoss()

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    refined_X_list = []
    final_preds_list = []

    for batch in dataloader:
        batch = batch.to(device)
        y_batch = batch.y.view(-1).float()  # [num_graphs]
        input_dim = batch.x.shape[1]

        n_graphs = y_batch.size(0)

        # initialize synthetic node features
        X_syn = torch.randn(n_graphs * n_syn, input_dim, requires_grad=True, device=device)

        # batch assignment for synthetic nodes
        batch_vec = torch.arange(n_graphs, device=device).repeat_interleave(n_syn)

        optimizer_X = torch.optim.SGD([X_syn], lr=lr)

        for epoch in range(epochs):
            optimizer_X.zero_grad()
            syn_batch = Batch(x=X_syn, batch=batch_vec, y=y_batch)
            preds = model(syn_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer_X.step()

        # save refined X and predictions as tensors
        refined_X_batch = torch.stack([X_syn[batch_vec == i] for i in range(n_graphs)], dim=0)  # [B, n_syn, input_dim]
        refined_X_list.append(refined_X_batch)
        final_preds_list.append(preds.detach())

    # concatenate batches along graph dimension
    refined_X = torch.cat(refined_X_list, dim=0)         # [total_graphs, n_syn, input_dim]
    final_preds = torch.cat(final_preds_list, dim=0)     # [total_graphs]

    return refined_X, final_preds

# -----------------------------
# 1️⃣ Build adjacency matrix
# -----------------------------
def batch_build_topk_adjacency(X: torch.Tensor, Z: int):
    """
    Build batched top-Z adjacency matrix using cosine similarity.

    Args:
        X: [B, N, d] node features
        Z: number of top connections per node

    Returns:
        A: [B, N, N] adjacency matrix
    """
    B, N, d = X.shape
    device = X.device

    X_norm = F.normalize(X, p=2, dim=2)          # [B, N, d]
    sim_matrix = X_norm @ X_norm.transpose(1, 2) # [B, N, N]

    topk_vals, topk_idx = torch.topk(sim_matrix, k=Z, dim=2)
    A = torch.zeros_like(sim_matrix)
    batch_idx = torch.arange(B, device=device)[:, None, None].expand(-1, N, Z)
    node_idx = torch.arange(N, device=device)[None, :, None].expand(B, -1, Z)
    A[batch_idx, node_idx, topk_idx] = 1

    return A

# -----------------------------
# 2️⃣ Build normalized Laplacian
# -----------------------------
def batch_compute_normalized_laplacian(A):
    """
    Compute normalized Laplacian for batched adjacency matrices A.
    A: [B, N, N]
    Returns L: [B, N, N]
    """
    device = A.device
    B, N, _ = A.shape

    # Degree matrix per batch
    deg = A.sum(dim=-1)  # [B, N]
    D_inv_sqrt = torch.pow(deg, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0  # handle isolated nodes

    # Compute normalized Laplacian
    I = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)  # [B, N, N]
    L = I - D_inv_sqrt[:, :, None] * A * D_inv_sqrt[:, None, :]  # broadcast correctly
    return L

# -----------------------------
# 3️⃣ Eigen-decomposition
# -----------------------------
def batch_compute_eigen(L: torch.Tensor, K1: int, K2: int):
    """
    Compute top K1+K2 eigenvectors and all eigenvalues for a batch of Laplacians.

    Args:
        L: [B, N, N] Laplacian matrix
        K1, K2: number of eigenvectors to keep

    Returns:
        U_all: [B, N, K1+K2] eigenvectors
        eigvals_all: [B, N] eigenvalues
    """
    B, N, _ = L.shape
    U_list = []
    eigvals_list = []
    for i in range(B):
        eigvals, eigvecs = torch.linalg.eigh(L[i])
        U_list.append(eigvecs[:, :K1+K2])
        eigvals_list.append(eigvals)
    U_all = torch.stack(U_list, dim=0)
    eigvals_all = torch.stack(eigvals_list, dim=0)
    return U_all, eigvals_all

# -----------------------------
# 4️⃣ Create dataloader
# -----------------------------
def create_dataloader(X, U, eigvals, batch_size=1, shuffle=False, save_dir=None):
    """
    Create and optionally save a PyG DataLoader from tensors.

    Args:
        X: [B, N, d] node features
        U: [B, N, K1+K2] eigenvectors
        eigvals: [B, N] eigenvalues
        batch_size: batch size
        shuffle: shuffle dataset
        save_dir: optional path to save the dataset (as .pt)

    Returns:
        DataLoader object
    """
    data_list = []

    B, N, d = X.shape
    for i in range(B):
        Xi = X[i]
        Ui = U[i]
        eigvali = eigvals[i]
        data = Data(x=Xi, u=Ui, eigval=eigvali)
        data_list.append(data)

    # ✅ Save dataset if a directory is given
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(data_list, os.path.join(save_dir, "dataset.pt"))
        print(f"✅ Dataset saved at {os.path.join(save_dir, 'dataset.pt')}")

    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    return dataloader




  
