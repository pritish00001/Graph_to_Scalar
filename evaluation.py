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

def build_train_loader(x_syn_final, u_syn_final, eigenval_list, y_list, k1_list, k2_list, threshold, batch_size):
    """
    Build a PyTorch Geometric DataLoader for synthetic graphs.

    Args:
        x_syn_final: list of node feature tensors
        u_syn_final: list of U tensors (eigenvectors)
        eigenval_list: list of eigenvalue tensors
        y_list: list of scalar targets
        k1_list: list of k1 values
        k2_list: list of k2 values
        threshold: threshold for edge creation
        batch_size: batch size for DataLoader

    Returns:
        train_loader: DataLoader for all synthetic graphs
    """
    data_list = []

    for i in range(len(x_syn_final)):
        Xi = x_syn_final[i]
        Ui = u_syn_final[i]
        evals = eigenval_list[i]
        k1 = k1_list[i]
        k2 = k2_list[i]

        adj = create_adj_prime(U=Ui, k1=k1, k2=k2, eigenvals=evals)
        edge_index, edge_weight = create_edge_index_using_adjacency_matrix(adj, threshold=threshold)

        data = Data(
            x=Xi,
            edge_index=edge_index,
            edge_weight=edge_weight.to(torch.float),
            y=torch.tensor([float(y_list[i])], dtype=torch.float)
        )
        data_list.append(data)

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

if __name__ == "__main__":
    default_config = {}
    x_syn_final = torch.load('x_syn_final.pt')
    u_syn_final = torch.load('u_syn_final.pt')
    eigen_vals_list = torch.load('eigen_vals_list.pt')
    X_real_list = torch.load('X_real_list.pt')
    y_list = torch.load('y_list.pt')
    X_real_test_list = torch.load('X_real_test_list.pt')
    y_test_list = torch.load('y_test_list.pt')
    edge_index_list = torch.load('edge_index_list.pt')
    edge_weight_list = torch.load('edge_weight_list.pt')
    edge_index_test_list = torch.load('edge_index_test_list.pt')
    edge_weight_test_list = torch.load('edge_weight_test_list.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k1_syn_list = []
    k2_syn_list = []
    r = default_config['r']
    for i in range(len(x_syn_final)):
        Xi = x_syn_final[i]
        n_syn, d = Xi.size()
        K1 = int(r*n_syn)
        K2 = int((1-r)*n_syn)
        k1_syn_list.append(K1)
        k2_syn_list.append(K2)

    # Build model
    hidden_dim = default_config['hidden_dim']
    dropout = default_config['dropout']
    lr_gnn = default_config['lr_gnn']
    threshold = default_config['threshold']
    batch_size = default_config['batch_size']

    epochs = 200

    in_channels = x_syn_final[0].shape[1]
    model_syn = GNN().to(device=device)
    gnn_syn_opt = optim.Adam(model_syn.parameters(), lr=lr_gnn)

    train_loader = build_train_loader(
        x_syn_final, u_syn_final, eigen_vals_list, y_list, k1_syn_list, k2_syn_list, threshold, batch_size
    )


    start_time = time.time()  # start timer

    for ep in range(epochs):
        print(f"{ep} done")
        l = gnn_regression_step(
            model_syn,
            train_loader,
            gnn_syn_opt
        )
        # print(l)

    end_time = time.time()  # end timer
    elapsed = end_time - start_time
    print(f"Total training time: {elapsed:.2f} seconds")
    print(f"Average time per epoch: {elapsed/epochs:.4f} seconds")

    y_pred_real = []

    for i in range(len(X_real_list)):
        Xi = X_real_list[i]
        edge_index = edge_index_list[i]
        edge_weight = edge_weight_list[i]
        edge_weight = edge_weight.to(torch.float)
        batch = torch.zeros(Xi.size(0), dtype=torch.long, device=Xi.device)

        # y_pred_test_real.append(model_syn(Xi, edge_index, batch, edge_weight))
        y_pred_real.append(model_syn(Xi, edge_index, batch))

    for i in range(len(y_pred_real)):
        print(f"pred:{y_pred_real[i].item()}" + 50*" " +f"true:{y_list[i]}" )

    mse_real, preds, trues = evaluate_syn(y_pred_real,y_list, device= 'cpu')

    print(f"Evaluaion MSE is : {mse_real}")

    save_scatter(trues, preds,
                'Train_Real_Graphs: True vs Predicted',
                'true_vs_pred_scatter_real.png')

    # List of saved files
    files = [
        'true_vs_pred_scatter_real.png'
    ]

    # Display each image
    for f in files:
        display(Image(filename=f))

    y_pred_test_real = []

    for i in range(len(X_real_test_list)):
        Xi = X_real_test_list[i]
        edge_index = edge_index_test_list[i]
        edge_weight = edge_weight_test_list[i]
        edge_weight = edge_weight.to(torch.float)
        batch = torch.zeros(Xi.size(0), dtype=torch.long, device=Xi.device)

        # y_pred_test_real.append(model_syn(Xi, edge_index, batch, edge_weight))
        y_pred_test_real.append(model_syn(Xi, edge_index, batch))

    for i in range(len(y_pred_test_real)):
        print(f"pred:{y_pred_test_real[i].item()}" + 50*" " +f"true:{y_list[i]}" )

    mse_real, preds, trues = evaluate_syn(y_pred_test_real,y_test_list, device= 'cpu')

    print(f"Evaluaion MSE is : {mse_real}")

    save_scatter(trues, preds,
                'Test_Real_Graphs: True vs Predicted',
                'true_vs_pred_test_scatter_real.png')

    # List of saved files
    files = [
        'true_vs_pred_test_scatter_real.png'
    ]

    # Display each image
    for f in files:
        display(Image(filename=f))

        # 0.29064813256263733