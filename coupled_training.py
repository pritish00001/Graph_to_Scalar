import torch
import time
import random
from typing import List
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from losses import Losses
from utils import create_adj_prime, create_edge_index_using_adjacency_matrix

def make_data(i, X_syn_list, U_syn_list, eigenval_list, y_tensor, K1_list, K2_list, threshold):
    """Constructs a PyG Data object for graph i (synthetic graph version)."""
    Xs, Us = X_syn_list[i], U_syn_list[i]
    yi = y_tensor[i].unsqueeze(0)  # graph label

    # adjacency and edge creation
    adj = create_adj_prime(U=Us, k1=K1_list[i], k2=K2_list[i], eigenvals=eigenval_list[i])
    edge_index, edge_weight = create_edge_index_using_adjacency_matrix(adj, threshold)

    return Data(
        x=Xs, edge_index=edge_index, edge_attr=edge_weight, y=yi,
        idx=i  # keep index so we can fetch X_real, U_real for Le
    )

def coupled_training(
    GNN_model: torch.nn.Module,
    X_real_list:  List[torch.Tensor],
    X_syn_list:   List[torch.nn.Parameter],
    U_real_list:  List[torch.Tensor],
    U_syn_list:   List[torch.nn.Parameter],
    eigenval_list: List[torch.Tensor],
    y_list:       List[float],
    x_optimizer:  torch.optim.Optimizer,
    u_optimizer:  torch.optim.Optimizer,
    gnn_optimizer: torch.optim.Optimizer,
    alpha: float,
    beta:  float,
    gamma: float,
    tau1:  int,
    tau2:  int,
    epochs: int,
    K1_list:    List[int],
    K2_list:    List[int],
    threshold: float = 1e-6,
    batch_size: int = 8,
    seed: int = 42
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(seed)
    torch.manual_seed(seed)

    # move model + data to device
    GNN_model = GNN_model.to(device)
    y_tensor = torch.tensor(y_list, dtype=torch.float32, device=device)

    N = len(X_real_list)
    le_total_all, lo_total_all, lr_total_all = 0.0, 0.0, 0.0
    total_count = 0

    for ep in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Build dataset fresh each epoch (because syn params change)
        dataset = [make_data(i, X_syn_list, U_syn_list, eigenval_list,
                             y_tensor, K1_list, K2_list, threshold) for i in range(N)]
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"=== Phase 1: Distillation (τ₁={tau1}) ===")
        GNN_model.eval()

        for t in range(tau1):
            print(f"\n-- Distillation Round {t+1}/{tau1} --")
            le_total, lo_total, lr_total = 0.0, 0.0, 0.0
            count = 0

            for batch in loader:
                batch = batch.to(device)
                x_optimizer.zero_grad()
                u_optimizer.zero_grad()

                # Compute regression loss
                y_pred = GNN_model(batch.x, batch.edge_index, batch.batch)
                Lr = Losses.regression_loss(y_pred, batch.y)

                # Collect tensors per graph in the batch for eigen & orthogonality
                batch_Xr = [X_real_list[i] for i in batch.idx.tolist()]
                batch_Xs = [X_syn_list[i] for i in batch.idx.tolist()]
                batch_Ur = [U_real_list[i] for i in batch.idx.tolist()]
                batch_Us = [U_syn_list[i] for i in batch.idx.tolist()]
                batch_K1 = [K1_list[i] for i in batch.idx.tolist()]
                batch_K2 = [K2_list[i] for i in batch.idx.tolist()]

                # Eigen alignment & orthogonality
                Le = Losses.eigen_alignment_loss(batch_Xr, batch_Ur, batch_Xs, batch_Us, batch_K1, batch_K2)
                Lo = Losses.orthogonality_loss(batch_Us)

                # Weighted total loss
                Lsyn = Losses.synthetic_graph_loss(Le, Lo, Lr, alpha, beta, gamma)
                Lsyn.backward()

                x_optimizer.step()
                u_optimizer.step()

                # Accumulate
                le_total += Le.item() * batch.num_graphs
                lo_total += Lo.item() * batch.num_graphs
                lr_total += Lr.item() * batch.num_graphs
                count += batch.num_graphs

            # Averages for this round
            avg_Le = le_total / count
            avg_Lo = lo_total / count
            avg_Lr = lr_total / count
            print(f"Summary — Round {t+1}/{tau1}: Avg Le={avg_Le:.6f}, Avg Lo={avg_Lo:.6f}, Avg Lr={avg_Lr:.6f}")

            le_total_all += le_total
            lo_total_all += lo_total
            lr_total_all += lr_total
            total_count += count

        epoch_time = time.perf_counter() - epoch_start
        print(f"\n=== Epoch {ep} Completed — Time: {epoch_time:.2f} sec ===\n")

    final_avg_Le = le_total_all / total_count
    final_avg_Lo = lo_total_all / total_count
    final_avg_Lr = lr_total_all / total_count

    print("\n=== Coupled Training Fully Complete ===")
    return final_avg_Le, final_avg_Lo, final_avg_Lr
