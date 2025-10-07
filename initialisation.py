import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import laplacian_eigen_decomposition, normalized_laplacian

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
        
    def forward(self, X):
        node_embeddings = self.mlp1(X)          # [N, 64]
        graph_embedding = node_embeddings.mean(dim=0)  # [64]
        y_hat = self.mlp2(graph_embedding)      # [output_dim]
        return y_hat
    
def train_models(X_real_list, y_list):
    
    all_models = []
    all_losses = []

    criterion = nn.MSELoss()

    for i in range(len(X_real_list)):
        # Create a new model for each graph
        model = GraphModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Prepare data
        X = X_real_list[i]
        y_true = torch.tensor([y_list[i]], dtype=torch.float32)
        
        # Quick training
        for _ in range(50):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y_true)
            loss.backward()
            optimizer.step()
        
        all_models.append(model)
        all_losses.append(loss.item())

    print(f"Trained {len(all_models)} models")
    return all_models, all_losses

def refine_all_X(all_models, X_list, y_list, n_list, R=0.7, r = 0.8, epochs=100, lr=0.1):
    """
    Args:
        all_models: list of trained GraphModel instances
        X_list: list of node feature tensors
        y_list: list of scalar targets (floats)
        n_list: list of number of nodes (ints)
        R: scaling factor for synthetic node count
        epochs: number of refinement steps
        lr: learning rate for updating X

    Returns:
        refined_X_list: list of refined synthetic node feature tensors
        final_preds: list of final predictions
    """
    criterion = nn.MSELoss()
    refined_X_list = []
    final_preds = []
    k1_list = []
    k2_list = []

    for i, model in enumerate(all_models):
        n = n_list[i]
        d = X_list[i].shape[1]      # feature dimension
        n_syn = int(R * n)

        K1 = int(r*n_syn)
        K2 = int((1-r)*n_syn)

        k1_list.append(K1)
        k2_list.append(K2)

        y_target = torch.tensor([y_list[i]], dtype=torch.float32)

        # initialize synthetic node features
        X_syn = torch.randn(n_syn, d, requires_grad=True)

        optimizer_X = torch.optim.SGD([X_syn], lr=lr)

        for epoch in range(epochs):
            optimizer_X.zero_grad()
            y_hat = model(X_syn)
            loss = criterion(y_hat, y_target)
            loss.backward()
            optimizer_X.step()

        refined_X_list.append(X_syn.detach())
        final_preds.append(y_hat.detach())

    return refined_X_list, k1_list, k2_list, final_preds

def build_adjacency(X, Z):
    """
    Args:
        X: Node feature matrix [N, d]
        Z: Number of top connections to keep per node (including self)
    Returns:
        A: Binary adjacency matrix [N, N]
    """
    # Normalize node features (for cosine similarity)
    X_norm = F.normalize(X, p=2, dim=1)  # [N, d]
    
    # Cosine similarity matrix
    sim_matrix = X_norm @ X_norm.T       # [N, N]
    # print("Similarity matrix:\n", sim_matrix)
    # Initialize adjacency
    A = torch.zeros_like(sim_matrix)
    
    # For each row, pick top Z indices (including self)
    topk = torch.topk(sim_matrix, k=Z, dim=1)
    rows = torch.arange(X.shape[0]).unsqueeze(1).expand(-1, Z)
    A[rows, topk.indices] = 1
    
    return A

if __name__ == "__main__":
  
    X_real_list = torch.load('X_real_list.pt')
    y_list = torch.load('y_list.pt')
    all_models = train_models(X_real_list, y_list)
    n_list = [X.shape[0] for X in X_real_list]
    X_syn_list, K1_list, K2_list, final_preds = refine_all_X(
    all_models, X_real_list, y_list, n_list=n_list, epochs=100, lr=0.05
)
    Z = 70           # Keep top 40 neighbors per node (including self)
    A_syn_list = []
    for i in range(len(X_syn_list)):
        X = X_syn_list[i]
        A = build_adjacency(X, Z)
        A_syn_list.append(A)

    print("Adjacency matrix:\n", A_syn_list[0].shape)
    print("Adjacency matrix:\n", A_syn_list[0])

    U_syn_list = []
    for i in range(len(A_syn_list)):
        L = normalized_laplacian(A_syn_list[i])
        U = laplacian_eigen_decomposition(L, K1_list[i], K2_list[i])
        U_syn_list.append(U)


    data_to_save = {
        "U_syn_list": U_syn_list,
        "X_syn_list": X_syn_list,
        "K1_list": K1_list,
        "K2_list": K2_list,
    }

    torch.save(data_to_save, "graph_data_1.pt")
    print("Data saved to graph_data_1.pt")