# enriched_data_safe.py
import os
import torch
import pickle
from typing import List, Optional
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from utils import (
    build_adjacency_from_data,
    create_adj_prime_from_eigendecomp,
    create_edge_index_using_adjacency_matrix,
    normalized_laplacian,
)

# -----------------------------------------------------------------------------
# Safe torch.load helper
# -----------------------------------------------------------------------------
_safe_globals = []
try:
    from torch_geometric.data.data import Data as PyGData
    _safe_globals.append(PyGData)
    if hasattr(torch_geometric.data.data, "DataEdgeAttr"):
        _safe_globals.append(getattr(torch_geometric.data.data, "DataEdgeAttr"))
except Exception:
    pass


def safe_torch_load(path: str):
    """
    Robust loader that tries several fallbacks:
      1) torch.load(path)
      2) torch.load(path, weights_only=False)
      3) torch.load(path, weights_only=False) inside add_safe_globals([...])
    """
    try:
        return torch.load(path)
    except Exception:
        try:
            return torch.load(path, weights_only=False)
        except Exception:
            if _safe_globals:
                with torch.serialization.add_safe_globals(_safe_globals):
                    return torch.load(path, weights_only=False)
            raise


# -----------------------------------------------------------------------------
# Dataset class that loads saved .pt graphs safely
# -----------------------------------------------------------------------------
class EnrichedDataset(torch.utils.data.Dataset):
    """Loads saved enriched Data objects from disk using safe_torch_load."""
    def __init__(self, file_paths: List[str]):
        self.file_paths = list(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = safe_torch_load(path)
        return data


# -----------------------------------------------------------------------------
# Main enrichment routine
# -----------------------------------------------------------------------------
def enrich_save_and_get_dataloader(
    src_loader,                     # original loader yielding PyG Batch objects
    out_dir: str,
    out_batch_size: int = 32,
    out_shuffle: bool = False,
    compute_eig: bool = True,
    batch_eig_max_N: int = 256,
    device: Optional[torch.device] = None,
    adj_threshold: float = 1e-6,
    save_prefix: str = "enriched",
    max_graphs: Optional[int] = None,
    num_workers_out: int = 0,
    show_progress: bool = True,
):
    """
    Process src_loader in batches, save enriched graphs to out_dir,
    and return a DataLoader that loads them lazily.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []
    total_saved = 0

    # Infer device if not provided
    if device is None:
        first_batch = next(iter(src_loader))
        device = getattr(first_batch.x, "device", torch.device("cpu"))

    iterator = tqdm(src_loader, desc="Processing batches") if show_progress else src_loader

    for batch in iterator:
        if max_graphs is not None and total_saved >= max_graphs:
            break

        # Build adj list for this batch
        _, adj_list = build_adjacency_from_data(batch, return_list=True, device=device)
        if len(adj_list) == 0:
            continue

        node_counts = [int(A.shape[0]) for A in adj_list]
        Nmax = max(node_counts)
        use_batched_eig = compute_eig and (Nmax <= batch_eig_max_N)

        batched_vals = batched_vecs = None
        if use_batched_eig:
            try:
                A_padded, _ = build_adjacency_from_data(batch, pad_to_max=True, device=device)
                L_batched = normalized_laplacian(A_padded)
                with torch.no_grad():
                    batched_vals, batched_vecs = torch.linalg.eigh(L_batched)
            except Exception as e:
                print("[WARN] Batched eigen failed; falling back:", e)
                use_batched_eig = False

        # Per-graph eigen
        per_vals, per_vecs = [None] * len(adj_list), [None] * len(adj_list)
        if compute_eig and not use_batched_eig:
            for i, Ai in enumerate(adj_list):
                if Ai.numel() == 0:
                    per_vals[i] = torch.zeros(0)
                    per_vecs[i] = torch.zeros((0, 0))
                    continue
                L = normalized_laplacian(Ai)
                with torch.no_grad():
                    per_vals[i], per_vecs[i] = torch.linalg.eigh(L)

        # Get individual Data objects
        data_list = getattr(batch, "to_data_list", lambda: None)()
        if not isinstance(data_list, list):
            data_list = None

        # Save enriched data
        for i, Ai in enumerate(adj_list):
            if max_graphs is not None and total_saved >= max_graphs:
                break

            if data_list is not None:
                data = data_list[i].clone() if hasattr(data_list[i], "clone") else data_list[i]
            else:
                data = Data(x=batch.x.clone().cpu())

            data.original_A = Ai.cpu()

            # Eigenpairs
            if use_batched_eig:
                vals_i = batched_vals[i, :node_counts[i]].cpu()
                vecs_i = batched_vecs[i, :node_counts[i], :node_counts[i]].cpu()
            elif compute_eig:
                vals_i, vecs_i = per_vals[i].cpu(), per_vecs[i].cpu()
            else:
                vals_i = vecs_i = None

            data.lap_eigenvals = vals_i
            data.lap_eigvecs = vecs_i

            # Adjacency prime
            if vals_i is not None and vecs_i is not None:
                try:
                    adj_prime = create_adj_prime_from_eigendecomp(
                        eigenvals=vals_i, eigenvecs=vecs_i, k1=node_counts[i], k2=0
                    )
                    if adj_prime is not None:
                        data.adj_prime = adj_prime.cpu()
                        ei, ew = create_edge_index_using_adjacency_matrix(adj_prime, threshold=adj_threshold)
                        data.adj_prime_edge_index = ei.cpu()
                        data.adj_prime_edge_weight = ew.cpu()
                except Exception as e:
                    print(f"[WARN] adj_prime failed for graph {i}: {e}")

            # Label
            if hasattr(data, "stiffness"):
                try:
                    data.y = torch.tensor(float(data.stiffness[0]), dtype=torch.float32)
                except Exception:
                    data.y = data.stiffness
            elif hasattr(data, "y") and isinstance(data.y, torch.Tensor):
                data.y = data.y.cpu()
            else:
                data.y = None

            try:
                data = data.to("cpu")
            except Exception:
                pass

            out_path = os.path.join(out_dir, f"{save_prefix}_{total_saved:06d}.pt")
            torch.save(data, out_path)
            saved_paths.append(out_path)
            total_saved += 1

    enriched_dataset = EnrichedDataset(saved_paths)
    enriched_loader = PyGDataLoader(
        enriched_dataset,
        batch_size=out_batch_size,
        shuffle=out_shuffle,
        num_workers=num_workers_out,
    )
    print(f"[INFO] Saved {total_saved} enriched graphs to {out_dir}.")
    return enriched_loader, saved_paths
