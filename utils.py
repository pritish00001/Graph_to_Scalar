import torch
from typing import List, Tuple, Optional

# Helper: get ptr (batch cumulative node counts) and device
def _get_ptr_and_device(data):
    # If data is a torch_geometric Batch it usually has .ptr (len B+1)
    if hasattr(data, "ptr"):
        ptr = data.ptr  # tensor
        device = ptr.device
    elif hasattr(data, "batch"):
        # build ptr from batch if ptr missing
        batch = data.batch
        counts = torch.bincount(batch)
        ptr = torch.cat([torch.tensor([0], device=batch.device), torch.cumsum(counts, dim=0)])
        device = batch.device
    else:
        # single graph
        ptr = None
        device = next((getattr(data, attr).device for attr in ("x","edge_index") if hasattr(data, attr)), torch.device("cpu"))
    return ptr, device

# Build adjacency for either a single graph Data or a Batch
def build_adjacency_from_data(
    data,
    edge_index: Optional[torch.LongTensor] = None,
    edge_weight: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    return_block_diag: bool = False,
    return_list: bool = False,
    pad_to_max: bool = False
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """
    Build adjacency matrices for graphs contained in `data` (Data or Batch).

    Returns:
      - If return_list: (None, list_of_adj) where list_of_adj is [A_0, A_1, ...]
      - If return_block_diag: (A_block_diag, None)
      - If pad_to_max: (A_padded, None) where A_padded shape [B, Nmax, Nmax]
      - Default: returns (A_block_diag, None)

    Parameters:
      data: PyG Data or Batch (must have edge_index). For single graph, treat as batch of size 1.
    """
    # infer edge_index and edge_weight from data if not provided
    global_edge_index = edge_index if edge_index is not None else getattr(data, "edge_index", None)
    global_edge_weight = edge_weight if edge_weight is not None else getattr(data, "edge_attr", None) or getattr(data, "edge_weight", None)

    ptr, inferred_device = _get_ptr_and_device(data)
    if device is None:
        device = inferred_device

    # single-graph (no ptr)
    if ptr is None:
        # number nodes
        num_nodes = getattr(data, "num_nodes", None)
        if num_nodes is None:
            # try infer from x
            if hasattr(data, "x"):
                num_nodes = data.x.size(0)
            else:
                raise ValueError("Cannot determine number of nodes for single graph data.")
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        if global_edge_index is not None:
            src = global_edge_index[0].long().to(device)
            dst = global_edge_index[1].long().to(device)
            if global_edge_weight is None:
                weights = torch.ones(src.size(0), device=device, dtype=torch.float32)
            else:
                weights = global_edge_weight.to(device=device, dtype=torch.float32)
            A[src, dst] = weights
        return A, None

    # Batched case: use ptr to slice nodes
    B = ptr.size(0) - 1
    adj_list = []
    max_nodes = 0

    ge = global_edge_index.to(device) if global_edge_index is not None else None
    gw = global_edge_weight.to(device) if global_edge_weight is not None else None

    for i in range(B):
        start = int(ptr[i].item())
        end = int(ptr[i + 1].item())
        n_i = end - start
        if n_i <= 0:
            adj_list.append(torch.zeros((0, 0), device=device))
            continue
        max_nodes = max(max_nodes, n_i)

        if ge is None:
            # no explicit edges -> zero adjacency
            A_i = torch.zeros((n_i, n_i), device=device)
        else:
            mask_src = (ge[0] >= start) & (ge[0] < end)
            mask_dst = (ge[1] >= start) & (ge[1] < end)
            mask = mask_src & mask_dst
            if mask.sum() == 0:
                A_i = torch.zeros((n_i, n_i), device=device)
            else:
                src = (ge[0, mask] - start).long()
                dst = (ge[1, mask] - start).long()
                if gw is None:
                    weights = torch.ones(src.size(0), device=device, dtype=torch.float32)
                else:
                    weights = gw[mask].to(device=device, dtype=torch.float32)
                A_i = torch.zeros((n_i, n_i), device=device)
                A_i[src, dst] = weights
        adj_list.append(A_i)

    if return_list:
        return None, adj_list

    if pad_to_max:
        # create tensor [B, Nmax, Nmax]
        Nmax = max_nodes
        padded = torch.zeros((B, Nmax, Nmax), device=device)
        for i, Ai in enumerate(adj_list):
            n = Ai.size(0)
            if n > 0:
                padded[i, :n, :n] = Ai
        return padded, None

    # block-diagonal adjacency (global adjacency for the batch)
    # torch.block_diag expects inputs on same device
    A_block = torch.block_diag(*adj_list) if len(adj_list) > 0 else torch.zeros((0,0), device=device)
    return A_block, None


# Degree matrix supports [N,N] or [B,N,N]
def degree_matrix(A: torch.Tensor) -> torch.Tensor:
    """
    If A is [N,N] returns [N,N] diagonal matrix.
    If A is [B,N,N] returns [B,N,N] diagonal matrices.
    """
    if A.dim() == 2:
        deg = A.sum(dim=1)
        return torch.diag(deg)
    elif A.dim() == 3:
        deg = A.sum(dim=2)  # [B,N]
        B, N = deg.shape
        D = torch.zeros((B, N, N), device=A.device, dtype=A.dtype)
        idx = torch.arange(N, device=A.device)
        D[:, idx, idx] = deg
        return D
    else:
        raise ValueError("A must be 2D or 3D tensor.")


def normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalized Laplacian:
      L = I - D^{-1/2} A D^{-1/2}
    Accepts A [N,N] or [B,N,N].
    """
    if A.dim() == 2:
        D = degree_matrix(A)
        deg = D.diag()
        deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
        return L
    elif A.dim() == 3:
        B, N, _ = A.shape
        Ls = []
        for i in range(B):
            Ai = A[i]
            Li = normalized_laplacian(Ai)
            Ls.append(Li)
        return torch.stack(Ls, dim=0)
    else:
        raise ValueError("A must be 2D or 3D tensor.")


def laplacian_eigen_decomposition(L: torch.Tensor, k1: int, k2: int):
    """
    For each adjacency L (2D or batched 3D), compute eigenvals/vecs and return:
      - eigenvals: [N] or [B, N]
      - eigenvecs: [N, N] or [B, N, N]
    Note: computing all eigenpairs may be costly for large graphs.
    """
    if L.dim() == 2:
        vals, vecs = torch.linalg.eigh(L)
        return vals, vecs
    elif L.dim() == 3:
        B, N, _ = L.shape
        vals_list = []
        vecs_list = []
        for i in range(B):
            vi, ui = torch.linalg.eigh(L[i])
            vals_list.append(vi)
            vecs_list.append(ui)
        vals = torch.stack(vals_list, dim=0)   # [B, N]
        vecs = torch.stack(vecs_list, dim=0)   # [B, N, N]
        return vals, vecs
    else:
        raise ValueError("L must be 2D or 3D.")


def create_adj_prime_from_eigendecomp(
    eigenvals: torch.Tensor,
    eigenvecs: torch.Tensor,
    k1: int,
    k2: int
) -> torch.Tensor:
    """
    Reconstruct A' = sum (1 - lambda_k) u_k u_k^T for required eigenpairs.
    Supports single-graph or batched:
     - eigenvals: [N] or [B,N]
     - eigenvecs: [N,N] or [B,N,N]
    Returns A_prime with same batching as inputs.
    """
    if eigenvecs.dim() == 2:
        N = eigenvecs.size(0)
        A_prime = torch.zeros((N, N), device=eigenvecs.device, dtype=eigenvecs.dtype)
        # smallest k1 from start, largest k2 from end
        for i in range(k1):
            lam = eigenvals[i]
            u = eigenvecs[:, i].unsqueeze(1)
            A_prime += (1 - lam) * (u @ u.t())
        for i in range(1, k2 + 1):
            lam = eigenvals[-i]
            u = eigenvecs[:, -i].unsqueeze(1)
            A_prime += (1 - lam) * (u @ u.t())
        return A_prime
    elif eigenvecs.dim() == 3:
        B, N, _ = eigenvecs.shape
        out = torch.zeros((B, N, N), device=eigenvecs.device, dtype=eigenvecs.dtype)
        for b in range(B):
            out[b] = create_adj_prime_from_eigendecomp(eigenvals[b], eigenvecs[b], k1, k2)
        return out
    else:
        raise ValueError("eigenvecs must be 2D or 3D.")


def create_edge_index_using_adjacency_matrix(
    adj: torch.Tensor,
    threshold: float = 1e-6,
    as_batch: bool = False,
    ptr: Optional[torch.Tensor] = None,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """
    Convert adjacency to edge_index and edge_weight.
    If adj is block-diagonal global adjacency for batch (2D) and `as_batch`==True
    you can pass ptr (graph ptr) to get per-graph local edge_index's.
    Returns (edge_index, edge_weight) in global indexing (default).
    """
    if adj.dim() == 2:
        mask = adj > threshold
        if mask.sum() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=adj.device), torch.empty((0,), device=adj.device)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj[mask].to(torch.float)
        return edge_index, edge_weight
    elif adj.dim() == 3:
        # return global concatenated edge_index with offsets
        B, N, _ = adj.shape
        all_src = []
        all_dst = []
        all_w = []
        for b in range(B):
            Ai = adj[b]
            mask = Ai > threshold
            if mask.sum() == 0:
                continue
            idx = mask.nonzero(as_tuple=False)
            src = idx[:, 0] + b * N
            dst = idx[:, 1] + b * N
            all_src.append(src)
            all_dst.append(dst)
            all_w.append(Ai[mask])
        if len(all_src) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=adj.device), torch.empty((0,), device=adj.device)
        src = torch.cat(all_src, dim=0)
        dst = torch.cat(all_dst, dim=0)
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = torch.cat(all_w, dim=0).to(adj.device)
        return edge_index, edge_weight
    else:
        raise ValueError("adj must be 2D (single) or 3D (batched)")


def unnormalize_adjacency(normalized_adj: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized_adj back to unnormalized adjacency using D (diagonal degree matrices).
    For single [N,N] or batched [B,N,N] forms.
    """
    if normalized_adj.dim() == 2:
        degree_vector = torch.diag(D)
        D_sqrt = torch.sqrt(degree_vector)
        D_row = D_sqrt.view(-1, 1)
        D_col = D_sqrt.view(1, -1)
        return D_row * normalized_adj * D_col
    elif normalized_adj.dim() == 3:
        B, N, _ = normalized_adj.shape
        out = torch.zeros_like(normalized_adj)
        for b in range(B):
            out[b] = unnormalize_adjacency(normalized_adj[b], D[b])
        return out
    else:
        raise ValueError("normalized_adj must be 2D or 3D.")
