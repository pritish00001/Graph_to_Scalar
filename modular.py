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
        self.graph_list = graph_list
        self.K1 = K1
        self.K2 = K2

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        graph = self.graph_list[idx]

        # Build adjacency and Laplacian
        n = graph.x.size(0)
        A = build_adjacency(graph.edge_index, num_nodes=n, edge_weight=getattr(graph, "edge_weight", None), device=graph.x.device)
        L = normalized_laplacian(A)

        # Eigendecomposition
        # eigenvals, eigenvecs = torch.linalg.eigh(L)
        eigenvals, eigenvecs = top_bottom_eigenpairs(L,self.K1,self.K2)

        # Optional sorting (ascending order)
        idx = torch.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        

        # Attach new attributes to Data object
        graph.eigenvecs = eigenvecs #same as u matrix
        graph.eigenvals = eigenvals
        graph.y = graph.stiffness  

        return graph 
    

# # Create dataset objects
# K1 = 10
# K2 = 10

# train_dataset = GraphEigenDataset(individual_list_of_training_graphs,K1,K2)

# Create DataLoaders
# BATCH_SIZE = 16

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)