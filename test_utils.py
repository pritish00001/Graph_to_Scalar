# test_utils.py
import torch
from load_data import load_train_test
from utils import (
    build_adjacency_from_data,
    normalized_laplacian,
    laplacian_eigen_decomposition,
    create_adj_prime_from_eigendecomp,
    create_edge_index_using_adjacency_matrix,
    unnormalize_adjacency,
)

def device_from_loader(loader):
    # infer device from first batch
    for b in loader:
        # handle PyG Batch or tensor
        if hasattr(b, "x"):
            return b.x.device
        elif isinstance(b, (tuple, list)) and isinstance(b[0], torch.Tensor):
            return b[0].device
        elif isinstance(b, torch.Tensor):
            return b.device
        else:
            return torch.device("cpu")
    return torch.device("cpu")


def test_utils_on_first_batch(train_loader, test_loader, device=None, max_graphs_to_eig=3):
    device = device or device_from_loader(train_loader)
    print("Using device:", device)

    # ---- get first batch ----
    batch = next(iter(train_loader))
    print("Batch type:", type(batch))

    # ---- 1) Build adjacencies: list, padded, block-diag ----
    print("\nBuilding adjacency representations...")
    _, adj_list = build_adjacency_from_data(batch, return_list=True)
    A_padded, _ = build_adjacency_from_data(batch, pad_to_max=True)
    A_block, _ = build_adjacency_from_data(batch)  # default is block-diagonal

    B = len(adj_list)
    Nmax = A_padded.shape[1] if A_padded is not None and A_padded.numel() > 0 else 0
    print(f"Batch contains {B} graphs. Nmax (padded) = {Nmax}")
    print("adj_list lengths:", [a.shape for a in adj_list][:10])
    print("A_padded shape:", None if A_padded is None else tuple(A_padded.shape))
    print("A_block shape:", tuple(A_block.shape) if A_block is not None else None)

    # ---- 2) Normalized Laplacian ----
    print("\nComputing normalized Laplacian (padded)...")
    if Nmax == 0:
        print("No nodes found in batch; skipping Laplacian/eig.")
        return
    L_batched = normalized_laplacian(A_padded)  # [B, Nmax, Nmax]
    print("L_batched shape:", tuple(L_batched.shape))

    # ---- 3) Eigendecomposition (LIMITED) ----
    print(f"\nRunning eigen-decomposition on up to {max_graphs_to_eig} graphs (to avoid heavy compute)...")
    num_eig_graphs = min(max_graphs_to_eig, B)
    # We will do per-graph eigendecomp on the first few graphs to be safe
    eigenvals_list = []
    eigenvecs_list = []
    for i in range(num_eig_graphs):
        Li = L_batched[i]
        # detect actual node count for this graph (might be less than Nmax)
        Ni = adj_list[i].shape[0]
        Li_small = Li[:Ni, :Ni] if Ni > 0 and Ni < Nmax else Li
        vals, vecs = torch.linalg.eigh(Li_small)  # safe for small Ni
        eigenvals_list.append(vals)
        eigenvecs_list.append(vecs)
        print(f"Graph {i}: Ni={Ni}, eigvals (first 5):", vals[:5].cpu().numpy())

    # ---- 4) Reconstruct A' for those graphs ----
    k1, k2 = 3, 0
    print(f"\nReconstructing A' using k1={k1}, k2={k2} for first {num_eig_graphs} graphs...")
    A_primes = []
    for i in range(num_eig_graphs):
        vals = eigenvals_list[i]
        vecs = eigenvecs_list[i]
        Aprime = create_adj_prime_from_eigendecomp(vals, vecs, k1=k1, k2=k2)
        A_primes.append(Aprime)
        print(f"Graph {i}: A' shape: {Aprime.shape}, A' sample (top-left 3x3):\n", Aprime[:3, :3].cpu().numpy())

    # ---- 5) Convert block adj back to edge_index/edge_weight ----
    print("\nConverting block-diagonal adjacency back to edge_index/edge_weight (threshold=1e-6)...")
    edge_index, edge_weight = create_edge_index_using_adjacency_matrix(A_block, threshold=1e-6)
    print("Recovered edge_index shape:", edge_index.shape, "edge_weight shape:", edge_weight.shape)
    print("Sample edges (first 10):")
    first_k = min(10, edge_index.shape[1])
    for idx in range(first_k):
        s = int(edge_index[0, idx].item())
        d = int(edge_index[1, idx].item())
        w = float(edge_weight[idx].item())
        print(f"  {idx}: {s} -> {d} (w={w:.6f})")

    # ---- 6) (Optional) Unnormalize adjacency example ----
    # Build degree diag for graph 0 and demonstrate unnormalize on normalized adjacency
    if adj_list and adj_list[0].numel() > 0:
        A0 = adj_list[0]
        D0 = torch.diag(A0.sum(dim=1))
        L0 = normalized_laplacian(A0)
        unA = unnormalize_adjacency(L0, D0)
        print("\nGraph 0: unnormalized adjacency reconstruction (sample 3x3):\n", unA[:3, :3].cpu().numpy())

    print("\nTest script finished successfully.")


if __name__ == "__main__":
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        torch.manual_seed(worker_seed)

    # load your data loaders (adapt this import to your code layout)
    train_loader, test_loader = load_train_test()

    # choose device (inferred from loader)
    dev = device_from_loader(train_loader)

    # Run the test
    test_utils_on_first_batch(train_loader, test_loader, device=dev, max_graphs_to_eig=3)
