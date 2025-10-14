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
from utils import *

warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

#change stiffness to y
# Custom Dataset Class
class GraphEigenDataset(Dataset):
    def __init__(self, graph_list, K1, K2, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.K1 = K1
        self.K2 = K2

        # Precompute eigenvals and eigenvecs
        self.graph_list = []
        for g in graph_list:
            n = g.x.size(0)
            # Build adjacency and Laplacian
            A = build_adjacency(
                g.edge_index, 
                num_nodes=n, 
                edge_weight=getattr(g, "edge_weight", None),
                device=g.x.device
            )
            L = normalized_laplacian(A)

            # Top-k and bottom-k eigenpairs
            eigenvals, eigenvecs = top_bottom_eigenpairs(L, self.K1, self.K2)

            # Optional sorting
            idx_sort = torch.argsort(eigenvals)
            eigenvals = eigenvals[idx_sort]
            eigenvecs = eigenvecs[:, idx_sort]

            # Attach computed attributes
            g.eigenvals = eigenvals
            g.eigenvecs = eigenvecs
            g.y = g.stiffness  # your target

            self.graph_list.append(g)

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        return self.graph_list[idx]

# # Create dataset objects
# K1 = 10
# K2 = 10

# train_dataset = GraphEigenDataset(individual_list_of_training_graphs,K1,K2)

# Create DataLoaders
# BATCH_SIZE = 16

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)