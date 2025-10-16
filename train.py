from losses import Losses
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from coupled_training import coupled_training, coupled_training_dataloaders
from models import GNN
from utils import create_adj_prime, create_edge_index_using_adjacency_matrix

def evaluate_mse(model, X_real_list, edge_index_list, edge_weight_list, y_list, device):

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(len(X_real_list)):
            x = X_real_list[i].to(device)
            edge_index = edge_index_list[i].to(device)
            edge_weight = edge_weight_list[i].to(device).to(torch.float)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
            # y_pred = model(x, edge_index, batch, edge_weight)
            y_pred = model(x, edge_index, batch)
            preds.append(y_pred.item())
            trues.append(y_list[i])
    preds = torch.tensor(preds, device=device)
    trues = torch.tensor(trues, device=device)
    return Losses.regression_loss(preds, trues).item(), preds, trues

def save_scatter(y_true, y_pred, title, filename, size=(6,6), dpi=300):
    plt.figure(figsize=size)
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"Saved → {filename}")

def train_with_config(config, X_real_list, edge_index_list, edge_weight_list, U_real_list, eigen_vals_list, y_list,
                      X_real_test_list, edge_index_test_list, edge_weight_test_list, y_test_list, device):
    """
    Train and evaluate the model with given hyperparameter configuration.
    Returns validation MSE history over epochs.
    """
    # Unpack fixed and current hyperparameters
    alpha, beta, gamma = config['alpha'], config['beta'], config['gamma']
    lr_x, lr_u, lr_gnn = config['lr_x'], config['lr_u'], config['lr_gnn']
    hidden_dim, dropout = config['hidden_dim'], config['dropout']
    r = config['r']
    Wg, Wf = config['Wg'], config['Wf']
    tau1, tau2 = config['tau1'], config['tau2']
    batch_size = config['batch_size']
    epochs = config['epochs']
    threshold = config.get('threshold', 1e-6)

    # # Build model
    # in_channels = X_real_list[0].shape[1]
    # model = GCNGraphRegressor(
    #     in_channels=in_channels,
    #     hidden_channels=hidden_dim,
    #     num_layers=2,
    #     dropout=dropout
    # ).to(device)



    # # Initialize synthetic parameters
    # X_syn_list, U_syn_list, K1_list, K2_list = [], [], [], []
    # for X_real, U_real in zip(X_real_list, U_real_list):
    #     n, d = X_real.size()
    #     n_syn = int(config['R'] * n)

    #     K1 = int(r*n_syn)
    #     K2 = int((1-r)*n_syn)

    #     k_total = K1 + K2
    #     K1_list.append(K1)
    #     K2_list.append(K2)
    #     X_syn_list.append(nn.Parameter(torch.randn(int(config['R'] * n), d, device=device)))
    #     U_syn_list.append(nn.Parameter(torch.randn(int(config['R'] * n), k_total, device=device)))

    
    checkpoint = torch.load("/content/drive/MyDrive/Example1/data/graph_data_1.pt")

    U_syn_list = checkpoint["U_syn_list"]
    X_syn_list = checkpoint["X_syn_list"]
    K1_list    = checkpoint["K1_list"]
    K2_list    = checkpoint["K2_list"]

    # Copy for training
    X_syn = [nn.Parameter(x.detach().clone().to(device)) for x in X_syn_list]
    U_syn = [nn.Parameter(u.detach().clone().to(device)) for u in U_syn_list]

    # Optimizers
    x_opt = optim.Adam(X_syn, lr=lr_x)
    u_opt = optim.Adam(U_syn, lr=lr_u)
    # gnn_opt = optim.Adam(model.parameters(), lr=lr_gnn)

    val_history = []
    train_history = []
    le_history = []
    lo_history = []
    lr_history = []


    for epoch in range(1, epochs + 1):
        # Coupled training for one epoch
        model = GNN().to(device)
        gnn_opt = optim.Adam(model.parameters(), lr=lr_gnn)

        le, lo, lr = coupled_training(
            GNN_model=model,
            X_real_list = X_real_list,
            X_syn_list = X_syn,
            U_real_list = U_real_list,
            U_syn_list = U_syn,
            eigenval_list = eigen_vals_list,
            y_list = y_list,
            x_optimizer=x_opt,
            u_optimizer=u_opt,
            gnn_optimizer=gnn_opt,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau1=tau1,
            tau2=tau2,
            epochs=1,
            K1_list=K1_list,
            K2_list=K2_list,
            threshold=threshold,
            batch_size=batch_size
        )
        # Evaluate
        edge_index_syn_list, edge_weight_syn_list = [], []
        for i in range(len(X_syn)):
            Xi = X_syn[i]
            Ui = U_syn[i]
            evals = eigen_vals_list[i] ##
            n_syn, d = Xi.size()
            r = r
            K1 = int(r*n_syn)
            K2 = int((1-r)*n_syn)

            adj = create_adj_prime(U=Ui, k1=K1, k2=K2, eigenvals=evals)

            edge_index, edge_weight = create_edge_index_using_adjacency_matrix(adj, 0.001)
            edge_weight = edge_weight.to(torch.float)

            edge_weight_syn_list.append(edge_weight)
            edge_index_syn_list.append(edge_index)

        mse_test, y_pred_test, y_true_test = evaluate_mse(model, X_real_test_list, edge_index_test_list, edge_weight_test_list, y_test_list, device)
        mse_train, y_pred_train, y_true_train = evaluate_mse(model, X_real_list, edge_index_list, edge_weight_list, y_list, device)
        mse_syn_train, y_pred_syn_train, y_true_syn_train = evaluate_mse(model, X_syn, edge_index_syn_list, edge_weight_syn_list, y_list, device)

        val_history.append(mse_test)
        train_history.append(mse_train)
        le_history.append(le)
        lo_history.append(lo)
        lr_history.append(lr)


        print(f"Epoch {epoch}/{epochs} | Val MSE: {mse_test:.6f} | Train Real Graphs MSE: {mse_train:.6f} | Train Syn Graphs MSE: {mse_syn_train:.6f}" )


    save_scatter(y_true_train, y_pred_train,
             'Train_Real_Graphs: True vs Predicted',
             'true_vs_pred_scatter_train.png')

    save_scatter(y_true_test,  y_pred_test,
                'Test: True vs Predicted',
                'true_vs_pred_scatter_test.png')

    save_scatter( y_true_syn_train, y_pred_syn_train,
                 "Train_Syn_Graphs: True vs Predicted",
                 'true_vs_pred_scatter_syn_train.png')


    return val_history, train_history, le_history, lo_history, lr_history, X_syn, U_syn


def tune_hyperparameter(param_name, param_values, default_config,
                        X_real_list, edge_index_list, edge_weight_list, U_real_list, eigen_vals_list, y_list,
                        X_real_test_list, edge_index_test_list, edge_weight_test_list, y_test_list, device):
    """
    Tune a single hyperparameter by varying its values while keeping others fixed.
    Collects histories and plots curves without retraining inside each plot.
    """
    all_histories = {}
    for val in param_values:
        cfg = default_config.copy()
        cfg[param_name] = val
        val_hist, train_hist, le_hist, lo_hist, lr_hist, X_syn, U_syn = train_with_config(
            cfg, X_real_list, edge_index_list, edge_weight_list, U_real_list, eigen_vals_list, y_list,
            X_real_test_list, edge_index_test_list, edge_weight_test_list, y_test_list, device
        )


        # print(f"x_syn_final ka shape: {len(x_syn_final)}, X_syn ka shape: {len(X_syn)}")
        # print(f"u_syn_final ka shape: {len(u_syn_final)}, U_syn ka shape: {len(U_syn)}")



        all_histories[val] = {
            'val': val_hist,
            'train': train_hist,
            'le': le_hist,
            'lo': lo_hist,
            'lr': lr_hist,
        }

    # === Plot Validation Curves ===
    plt.figure()
    for val, history in all_histories.items():
        plt.plot(range(1, len(history['val'])+1), history['val'], label=f"{param_name}={val}")
    plt.xlabel('Epoch')
    plt.ylabel('Validation MSE')
    plt.title(f'Validation Curves: Tuning {param_name}')
    plt.legend()
    plt.show()

    # === Plot Training Curves ===
    plt.figure()
    for val, history in all_histories.items():
        plt.plot(range(1, len(history['train'])+1), history['train'], label=f"{param_name}={val}")
    plt.xlabel('Epoch')
    plt.ylabel('Training MSE')
    plt.title(f'Training Curves: Tuning {param_name}')
    plt.legend()
    plt.show()

    # === Combined Train & Validation Curves ===
    plt.figure(figsize=(8, 5))
    for val, history in all_histories.items():
        epochs = range(1, len(history['val'])+1)
        plt.plot(epochs, history['val'], label=f"Val {param_name}={val}")
        plt.plot(epochs, history['train'], '--', label=f"Train {param_name}={val}")
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'Train & Val Curves: Tuning {param_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Optional: Plot Le, Lo, Lr Histories ===
    for loss_name in ['le', 'lo', 'lr']:
        plt.figure(figsize=(10, 5))
        for val, history in all_histories.items():
            plt.plot(range(1, len(history[loss_name]) + 1), history[loss_name], label=f"{param_name}={val}")
        plt.xlabel('Epoch')
        plt.ylabel(f'{loss_name.upper()} Loss')
        plt.title(f'{loss_name.upper()} Loss over Epochs: Tuning {param_name}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return X_syn, U_syn

# if __name__ == '__main__':
   
#     default_config = {
#         'alpha': 0.01 , 'beta': 1 , 'gamma': 1,
#         'lr_x': 0.1, 'lr_u': 0.1, 'lr_gnn': 0.001,
#         'hidden_dim': 64, 'dropout': 0.5,
#         'r':0.8, 'R': 0.7,
#         'Wg': 0, 'Wf': 0,
#         'tau1': 1, 'tau2': 10,
#         'batch_size': 64,
#         'epochs': 250, # for 50 = 250, 100 == 500
#         'threshold': 1e-3
#     }
#     x_syn_final = torch.load('x_syn_final.pt')
#     u_syn_final = torch.load('u_syn_final.pt')
#     eigen_vals_list = torch.load('eigen_vals_list.pt')
#     X_real_list = torch.load('X_real_list.pt')
#     U_real_list = torch.load('U_real_list.pt')
#     y_list = torch.load('y_list.pt')
#     X_real_test_list = torch.load('X_real_test_list.pt')
#     y_test_list = torch.load('y_test_list.pt')
#     edge_index_list = torch.load('edge_index_list.pt')
#     edge_weight_list = torch.load('edge_weight_list.pt')
#     edge_index_test_list = torch.load('edge_index_test_list.pt')
#     edge_weight_test_list = torch.load('edge_weight_test_list.pt')
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # alpha [0.1,0.01,0.001]
#     # beta [1,0.1,0.01]
#     # gamma [1000,100,10]

#     # lr_x = [0.1,0.01,0.001]
#     # lr_u = [0.1,0.01,0.001]
#     # lr_gnn = [0.01,0.001,0.0001]

#     # Choose the hyperparameter to tune and its candidate values
#     param = 'alpha'
#     values = [0.1]

#     # Call the tuning function (ensure data and device are defined)
#     x_syn_final, u_syn_final = tune_hyperparameter(param, values, default_config,
#                         X_real_list,edge_index_list, edge_weight_list, U_real_list, eigen_vals_list, y_list,
#                     X_real_test_list, edge_index_test_list, edge_weight_test_list, y_test_list, device)
    
import matplotlib.pyplot as plt
import torch

def evaluate_mse_dataloader(model, dataloader, device):
    """
    Evaluate model MSE using a PyG DataLoader.
    Returns: (mse_loss, preds_tensor, trues_tensor)
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            y_pred = model(batch.x, batch.edge_index, batch.batch)
            preds.append(y_pred.view(-1))
            trues.append(batch.y.view(-1))

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    mse_loss = Losses.regression_loss(preds, trues).item()
    return mse_loss, preds, trues


def save_scatter(y_true, y_pred, title, filename, size=(6, 6), dpi=300):
    """
    Save a scatter plot comparing true vs predicted values.
    """
    # Detach and move to CPU if tensors
    if torch.is_tensor(y_true): y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.detach().cpu().numpy()

    plt.figure(figsize=size)
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"✅ Saved scatter plot → {filename}")

def train_with_config_dataloader(
    config,
    train_loader,
    test_loader,
    synthetic_graph_list,
    device,
    K1,
    K2
):
    """
    Train and evaluate the model with dataloaders and synthetic data.
    """
    # Unpack hyperparams
    alpha, beta, gamma = config['alpha'], config['beta'], config['gamma']
    lr_x, lr_u, lr_gnn = config['lr_x'], config['lr_u'], config['lr_gnn']
    hidden_dim, dropout = config['hidden_dim'], config['dropout']
    tau1, tau2 = config['tau1'], config['tau2']
    batch_size = config['batch_size']
    epochs = config['epochs']
    threshold = config.get('threshold', 1e-6)

    # Make synthetic node & eigen matrices learnable
    for data in synthetic_graph_list:
        data.x = nn.Parameter(data.x.clone().detach().requires_grad_(True))
        data.u = nn.Parameter(data.u.clone().detach().requires_grad_(True))

    # Optimizers
    x_optimizer = torch.optim.Adam([d.x for d in synthetic_graph_list], lr=lr_x)
    u_optimizer = torch.optim.Adam([d.u for d in synthetic_graph_list], lr=lr_u)

    # Histories
    val_history, train_history, le_history, lo_history, lr_history = [], [], [], [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Reinitialize GNN each epoch (optional)
        GNN_model = GNN().to(device)
        gnn_optimizer = torch.optim.Adam(GNN_model.parameters(), lr=lr_gnn)

        # Coupled training step using dataloaders
        le, lo, lr = coupled_training_dataloaders(
            GNN_model=GNN_model,
            train_dataset=train_loader.dataset,
            synthetic_graph_list=synthetic_graph_list,
            x_optimizer=x_optimizer,
            u_optimizer=u_optimizer,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau1=tau1,
            epochs=1,
            K1=K1,
            K2=K2,
            batch_size=batch_size,
            threshold=threshold,
        )

        # Evaluate
        mse_test, y_pred_test, y_true_test = evaluate_mse_dataloader(GNN_model, test_loader, device)
        mse_train, y_pred_train, y_true_train = evaluate_mse_dataloader(GNN_model, train_loader, device)

        val_history.append(mse_test)
        train_history.append(mse_train)
        le_history.append(le)
        lo_history.append(lo)
        lr_history.append(lr)

        print(f"Val MSE: {mse_test:.6f} | Train MSE: {mse_train:.6f}")

    # Save scatter plots
    save_scatter(y_true_train, y_pred_train, 'Train: True vs Predicted', 'scatter_train.png')
    save_scatter(y_true_test, y_pred_test, 'Test: True vs Predicted', 'scatter_test.png')

    return val_history, train_history, le_history, lo_history, lr_history


def tune_hyperparameter_dataloader(
    param_name,
    param_values,
    default_config,
    train_loader,
    test_loader,
    synthetic_graph_list,
    device,
    K1,
    K2
):
    """
    Tune a single hyperparameter by varying its values while keeping others fixed.
    Uses dataloaders and synthetic graph list instead of individual graph lists.
    """
    all_histories = {}
    best_val_mse = float('inf')
    best_val = None
    best_X_syn, best_U_syn = None, None

    for val in param_values:
        print(f"\n🔹 Tuning {param_name} = {val}")
        cfg = default_config.copy()
        cfg[param_name] = val

        (
            val_hist,
            train_hist,
            le_hist,
            lo_hist,
            lr_hist
        ) = train_with_config_dataloader(
            config=cfg,
            train_loader=train_loader,
            test_loader=test_loader,
            synthetic_graph_list=synthetic_graph_list,
            device=device,
            K1=K1,
            K2=K2
        )

        all_histories[val] = {
            'val': val_hist,
            'train': train_hist,
            'le': le_hist,
            'lo': lo_hist,
            'lr': lr_hist,
        }

        # Track best validation MSE (last epoch)
        final_val_mse = val_hist[-1]
        if final_val_mse < best_val_mse:
            best_val_mse = final_val_mse
            best_val = val
            best_X_syn = [d.x.detach().cpu() for d in synthetic_graph_list]
            best_U_syn = [d.u.detach().cpu() for d in synthetic_graph_list]

    # === Plot Validation Curves ===
    plt.figure(figsize=(8, 5))
    for val, history in all_histories.items():
        plt.plot(range(1, len(history['val'])+1), history['val'], label=f"{param_name}={val}")
    plt.xlabel('Epoch')
    plt.ylabel('Validation MSE')
    plt.title(f'Validation Curves: Tuning {param_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot Training Curves ===
    plt.figure(figsize=(8, 5))
    for val, history in all_histories.items():
        plt.plot(range(1, len(history['train'])+1), history['train'], label=f"{param_name}={val}")
    plt.xlabel('Epoch')
    plt.ylabel('Training MSE')
    plt.title(f'Training Curves: Tuning {param_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Combined Train & Validation Curves ===
    plt.figure(figsize=(8, 5))
    for val, history in all_histories.items():
        epochs = range(1, len(history['val'])+1)
        plt.plot(epochs, history['val'], label=f"Val {param_name}={val}")
        plt.plot(epochs, history['train'], '--', label=f"Train {param_name}={val}")
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'Train & Val Curves: Tuning {param_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Optional: Plot Le, Lo, Lr Histories ===
    for loss_name in ['le', 'lo', 'lr']:
        plt.figure(figsize=(10, 5))
        for val, history in all_histories.items():
            plt.plot(
                range(1, len(history[loss_name]) + 1),
                history[loss_name],
                label=f"{param_name}={val}"
            )
        plt.xlabel('Epoch')
        plt.ylabel(f'{loss_name.upper()} Loss')
        plt.title(f'{loss_name.upper()} Loss over Epochs: Tuning {param_name}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"\n✅ Best {param_name} = {best_val} (Val MSE = {best_val_mse:.6f})")

    return best_X_syn, best_U_syn, all_histories
