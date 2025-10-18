import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from IPython.display import Image, display
from losses import Losses
from models import GNN
from train import save_scatter
import time
import torch.optim as optim
from utils import create_adj_prime, create_edge_index_using_adjacency_matrix

def build_train_loader(synthetic_graph_list, train_dataset, K1, K2, threshold, batch_size):
    """
    Build a PyTorch Geometric DataLoader for synthetic graphs using synthetic graph list
    and a precomputed GraphEigenDataset for eigenvalues, eigenvectors, and targets.

    Args:
        synthetic_graph_list: list of synthetic graphs (each with g.x and g.u)
        train_dataset: GraphEigenDataset object (provides eigenvals, eigenvecs, y)
        K1, K2: constants for eigen decomposition
        threshold: threshold for edge creation
        batch_size: DataLoader batch size

    Returns:
        train_loader: DataLoader for all synthetic graphs
    """
    data_list = []

    for i, g_syn in enumerate(synthetic_graph_list):
        # Get corresponding graph from train dataset
        g_real = train_dataset.get(i)
        
        # Extract synthetic features and eigenvectors
        X_syn = g_syn.x
        U_syn = g_syn.u

        # Extract eigenvals and target from the real dataset
        eigenvals = g_real.eigenvals
        y_target = torch.tensor([float(g_real.y)], dtype=torch.float)

        # Build adjacency and edge index
        adj = create_adj_prime(U=U_syn, k1=K1, k2=K2, eigenvals=eigenvals)
        edge_index, edge_weight = create_edge_index_using_adjacency_matrix(adj, threshold=threshold)

        # Build graph data object
        data = Data(
            x=X_syn,
            edge_index=edge_index,
            edge_weight=edge_weight.to(torch.float),
            y=y_target
        )
        data_list.append(data)

    # Batch safely if dataset smaller than batch size
    train_loader = DataLoader(data_list, batch_size=min(batch_size, len(data_list)), shuffle=True)
    return train_loader

def gnn_regression_step(
    GNN_model,
    train_loader,
    optimizer
):
    GNN_model.train()
    # Step 3: Standard PyG training loop
    total_loss = 0.0
    x = 0.0
    for batch in train_loader:
        batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer.zero_grad()
        # pred = GNN_model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        pred = GNN_model(batch.x, batch.edge_index, batch.batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        x += batch.num_graphs


    return total_loss / x

def evaluate_syn(preds, trues, device):
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, device=device)
    else:
        preds = preds.to(device)

    if not isinstance(trues, torch.Tensor):
        trues = torch.tensor(trues, device=device)
    else:
        trues = trues.to(device)

    # detach before converting to scalar
    loss = Losses.regression_loss(preds, trues).detach().item()

    return loss, preds, trues

def train_synthetic_gnn(
    model,
    optimizer,
    synthetic_graph_list,
    train_dataset,
    default_config,
    K1,
    K2,
):
    """
    Full training pipeline for synthetic GNN model using synthetic graphs
    and precomputed real dataset eigen information.

    Args:
        model: an initialized (and possibly pre-trained) GNN model
        optimizer: optimizer for the model
        synthetic_graph_list: list of synthetic graphs (each with g.x, g.u)
        train_dataset: GraphEigenDataset containing eigenvals, eigenvecs, y
        default_config: dict with keys ['threshold', 'batch_size', 'epochs']
        K1, K2: constants for eigen decomposition
        device: 'cuda' or 'cpu'

    Returns:
        model: trained GNN model
        train_loader: DataLoader used for training
    """

    threshold = default_config['threshold']
    batch_size = default_config['batch_size']
    epochs = default_config.get('epochs', 200)

    # ===== Build train DataLoader =====
    train_loader = build_train_loader(
        synthetic_graph_list=synthetic_graph_list,
        train_dataset=train_dataset,
        K1=K1,
        K2=K2,
        threshold=threshold,
        batch_size=batch_size
    )

    # ===== Training Loop =====
    start_time = time.time()
    print("\n=== Starting Synthetic GNN Training ===")

    for ep in range(epochs):
        loss_val = gnn_regression_step(model, train_loader, optimizer)
        print(f"Epoch [{ep+1}/{epochs}] | Loss: {loss_val:.6f}")

    # ===== Timing Summary =====
    end_time = time.time()
    elapsed = end_time - start_time
    print("\n=== Training Complete ===")
    print(f"Total training time: {elapsed:.2f} seconds")
    print(f"Average time per epoch: {elapsed/epochs:.4f} seconds\n")

    return model, train_loader

def evaluate_trained_model_on_dataset(
    model,
    dataset,
    device,
    scatter_title='True vs Predicted',
    scatter_filename='true_vs_pred_scatter.png'
):
    """
    Evaluate a trained GNN model on a given GraphEigenDataset.

    Args:
        model: trained GNN model
        dataset: GraphEigenDataset object containing graph_list with .x, .edge_index, .edge_weight, .y
        device: torch device ('cpu' or 'cuda')
        scatter_title: title for scatter plot
        scatter_filename: filename to save the scatter plot image

    Returns:
        mse: mean squared error
        preds: predicted values
        trues: true values
    """
    model.eval()
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for g in dataset.graph_list:
            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            edge_weight = getattr(g, "edge_weight", None)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

            # Forward pass
            y_pred = model(x, edge_index, batch)

            y_pred_list.append(y_pred.cpu())
            y_true_list.append(float(g.y))

    # Print predictions vs true values
    print("\n=== Predictions vs Ground Truth ===")
    for pred, true_val in zip(y_pred_list, y_true_list):
        print(f"pred: {pred.item():.6f}" + 40*" " + f"true: {true_val:.6f}")

    # Compute evaluation metrics
    mse, preds, trues = evaluate_syn(y_pred_list, y_true_list, device='cpu')
    print(f"\nEvaluation MSE: {mse:.6f}")

    # Save scatter plot
    save_scatter(trues, preds, scatter_title, scatter_filename)

    # Display saved image
    display(Image(filename=scatter_filename))

    return mse, preds, trues

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    default_config = {
        'hidden_dim': 128,
        'dropout': 0.1,
        'lr_gnn': 1e-3,
        'threshold': 0.05,
        'batch_size': 8,
        'epochs': 200
    }

    K1 = 10
    K2 = 10

    # === Prepare dataset and synthetic graphs ===
    # synthetic_graph_list = [...]
    # train_dataset = GraphEigenDataset(real_graph_list, K1, K2)

    # === Initialize model and optimizer ===
    model = GNN(hidden_dim=default_config['hidden_dim'], dropout=default_config['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=default_config['lr_gnn'])
    synthetic_graph_list = []  # Placeholder for synthetic graphs
    train_dataset = None  # Placeholder for the real dataset
    # === Train ===
    trained_model, train_loader = train_synthetic_gnn(
        model=model,
        optimizer=optimizer,
        synthetic_graph_list=synthetic_graph_list,
        train_dataset=train_dataset,
        default_config=default_config,
        K1=K1,
        K2=K2,
        device=device,
    )

    # === Evaluate ===
    mse_real, preds, trues = evaluate_trained_model_on_dataset(
        model=trained_model,
        dataset=train_dataset,
        device=device,
        scatter_title='Train Real Graphs: True vs Predicted',
        scatter_filename='true_vs_pred_scatter_real.png'
    )

    print(f"\n✅ Final Evaluation Complete! MSE = {mse_real:.6f}")
    display(Image(filename='true_vs_pred_scatter_real.png'))