import torch
from typing import Tuple
import os
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import scipy.sparse as sp
from scipy.sparse.linalg import lobpcg
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="scipy")



def build_adjacency(edge_index, num_nodes, edge_weight=None, device='cuda'):
    """
    Build dense adjacency matrix A from edge_index and optional edge_weight.
    """
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    src, dst = edge_index[0].long(), edge_index[1].long()

    if edge_weight is None:
        edge_weight = torch.ones(src.size(0), dtype=torch.float32, device=device)
    else:
        edge_weight = edge_weight.to(dtype=torch.float32, device=device)

    A[src, dst] = edge_weight
    return A

def degree_matrix(A):
    """
    Compute degree matrix D = diag(sum of rows of A).

    Parameters
    ----------
    A : Tensor [N, N] — Adjacency matrix

    Returns
    -------
    D : Tensor [N, N] — Diagonal degree matrix
    """
    deg = A.sum(dim=1)
    return torch.diag(deg)

def normalized_laplacian(A):
    """
    Compute symmetric normalized Laplacian:
    L = I - D^{-1/2} * A * D^{-1/2}

    Parameters
    ----------
    A : Tensor [n, n] — adjacency matrix

    Returns
    -------
    L : Tensor [n, n] — normalized Laplacian
    """
    D = degree_matrix(A)
    deg = D.diag()
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    I = torch.eye(A.size(0), device=A.device)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def laplacian_eigen_decomposition(L, k1, k2):
    """
    Compute the first k1 smallest and last k2 largest eigenvectors of Laplacian L.

    Parameters
    ----------
    L : Tensor [n, n] (symmetric)
    k1 : int — number of smallest eigenvectors
    k2 : int — number of largest eigenvectors

    Returns
    -------
    U : Tensor [n, k1 + k2]
        Concatenated eigenvectors: [smallest-k1 | largest-k2]
    """
    eigenvals, eigenvecs = torch.linalg.eigh(L)  # sorted in ascending order
    U_small = eigenvecs[:, :k1]
    if k2 > 0:
        U_large = eigenvecs[:, -k2:]
        U = torch.cat([U_small, U_large], dim=1)
    else:
        U = U_small
    return U

def create_adj_prime(U: torch.Tensor, k1: int, k2: int, eigenvals: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs an adjacency matrix A' using the formula:
    A' = Σ (1 - λ_k) * u_k * u_k^T for top-k1 and bottom-k2 eigenpairs.

    Args:
        U (torch.Tensor): [N, N] matrix of eigenvectors.
        k1 (int): Number of top eigenvectors to use.
        k2 (int): Number of bottom eigenvectors to use.
        eigenvals (torch.Tensor): [N] tensor of eigenvalues.

    Returns:
        A_prime (torch.Tensor): [N, N] reconstructed adjacency matrix.
    """
    N = U.size(0)
    A_prime = torch.zeros((N, N), device=U.device)

    # Top-k1 eigenvectors (largest eigenvalues)
    for i in range(k1):
        λ = eigenvals[i]
        u = U[:, i].unsqueeze(1)  # [N, 1]
        A_prime += (1 - λ) * (u @ u.t())  # u * u.T

    # Bottom-k2 eigenvectors (smallest eigenvalues)
    for i in range(1, k2 + 1):
        λ = eigenvals[-i]
        u = U[:, -i].unsqueeze(1)
        A_prime += (1 - λ) * (u @ u.t())

    return A_prime

def create_edge_index_using_adjacency_matrix(
    adj: torch.Tensor,
    threshold: float = 0.01
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """
    Converts an adjacency matrix into edge_index and edge_weight format for PyTorch Geometric.

    Args:
        adj (torch.Tensor): [N, N] weighted adjacency matrix.
        threshold (float): Minimum edge weight to keep.

    Returns:
        edge_index (torch.LongTensor): [2, num_edges]
        edge_weight (torch.Tensor): [num_edges]
    """
    # Flattened mask for valid edges
    mask = adj > threshold

    # Get edge indices (i,j) where weight > threshold
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, num_edges]

    # Get corresponding edge weights
    edge_weight = adj[mask]  # [num_edges]
    edge_weight = edge_weight.to(torch.float)

    return edge_index, edge_weight

def unnormalize_adjacency(normalized_adj, D):
    """
    Given normalized adjacency and degree matrix D (square diagonal), recover the unnormalized adj.
    Assumes D is [N x N] diagonal matrix.
    """
    degree_vector = torch.diag(D)  # extract degree vector of shape [N]
    D_sqrt = torch.sqrt(degree_vector)

    D_row = D_sqrt.view(-1, 1)  # shape [N, 1]
    D_col = D_sqrt.view(1, -1)  # shape [1, N]

    unnormalized_adj = D_row * normalized_adj * D_col  # shape [N, N]
    return unnormalized_adj

def top_bottom_eigenpairs(L, k1=10, k2=10):
    # Convert to scipy sparse matrix
    if isinstance(L, torch.Tensor):
        L = L.detach().cpu()
        L = sp.csr_matrix(L)

    n = L.shape[0]
    eigenvals = []
    eigenvecs = []

    # Bottom k2 (smallest eigenvalues)
    if k2 > 0:
        X = np.random.randn(n, k2)
        vals, vecs = lobpcg(L, X, largest=False, maxiter=50, tol=1e-4)
        eigenvals.append(vals)
        eigenvecs.append(vecs)

    # Top k1 (largest eigenvalues)
    if k1 > 0:
        X = np.random.randn(n, k1)
        vals, vecs = lobpcg(L, X, largest=True, maxiter=50, tol=1e-4)
        eigenvals.append(vals)
        eigenvecs.append(vecs)

    # Concatenate
    eigenvals = np.concatenate(eigenvals)
    eigenvecs = np.concatenate(eigenvecs, axis=1)

    return torch.from_numpy(eigenvals), torch.from_numpy(eigenvecs)
