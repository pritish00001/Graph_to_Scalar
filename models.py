import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from torch_geometric.nn import Linear, SAGEConv, global_mean_pool

class GNN(torch.nn.Module):
    '''
    Graph Neural Network
    '''
    def __init__(self, N_fl1 = 32, N_mpl = 64, N_fl2 = 64, N_fl3 = 16):
        super(GNN, self).__init__()
        self.pre = Linear(5, N_fl1)
        self.conv1 = SAGEConv(N_fl1, N_mpl, normalize=True)
        self.conv2 = SAGEConv(N_mpl, N_mpl, normalize=True)
        self.post1 = Linear(N_mpl, N_fl2)
        self.post2 = Linear(N_fl2, N_fl3)
        self.out = Linear(N_fl3, 1)

    def forward(self,
            x: torch.Tensor,
            edge_index: torch.LongTensor,
            batch: torch.LongTensor
        ):

        # Pre Processing Linear Layer
        x = F.relu(self.pre(x))
        # 1. Obtain node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # 2. Readout layer
        x = global_mean_pool(x, batch)
        # 3. Apply Fully Connected Layers
        x = F.relu(self.post1(x))
        x = F.relu(self.post2(x))
        # print(x.size())
        x = self.out(x)
        return x.squeeze(-1)