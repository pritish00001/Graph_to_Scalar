import torch
from sklearn.preprocessing import StandardScaler

def collect_all_batches(loader):
    """Concatenate all batches from DataLoader or PyG DataLoader."""
    X_all, y_all = [], []
    first_batch = None

    for batch in loader:
        first_batch = batch if first_batch is None else first_batch

        # 🧩 Case 1: PyG DataBatch
        if hasattr(batch, "x"):  # PyTorch Geometric Batch
            X = batch.x
            y = getattr(batch, "y", None)

        # 🧩 Case 2: (X, y) tuple
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, y = batch

        # 🧩 Case 3: Only features
        else:
            X, y = batch, None

        X_all.append(X)
        if y is not None:
            y_all.append(y)

    # Handle concatenation depending on data type
    if hasattr(first_batch, "x"):  # PyG data
        X_all = torch.cat(X_all, dim=0)  # node features
        y_all = torch.cat(y_all, dim=0) if y_all else None
    else:
        X_all = torch.cat(X_all, dim=0)
        y_all = torch.cat(y_all, dim=0) if y_all else None

    return X_all, y_all


def preprocess_features(X, strategy='mean', scaler=None, device=None):
    """Device-safe StandardScaler (CPU or CUDA)."""
    if device is None:
        device = X.device if isinstance(X, torch.Tensor) else 'cpu'

    X_cpu = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cpu)
    else:
        X_scaled = scaler.transform(X_cpu)

    return torch.tensor(X_scaled, dtype=torch.float32, device=device), scaler


def scale_dataloaders(train_loader, test_loader, device='cpu'):
    """
    Works for both Tensor and PyG DataLoaders.
    Fits scaler on training node/features and scales test set accordingly.
    """
    # 1️⃣ Collect all features
    X_train, y_train = collect_all_batches(train_loader)
    X_test, y_test = collect_all_batches(test_loader)

    # 2️⃣ Fit and scale
    X_train_scaled, scaler = preprocess_features(X_train, scaler=None, device=device)
    X_test_scaled, _ = preprocess_features(X_test, scaler=scaler, device=device)

    # 3️⃣ If it's PyG data, replace .x
    first_train_batch = next(iter(train_loader))
    if hasattr(first_train_batch, "x"):  # PyG Dataset
        i_train, i_test = 0, 0
        scaled_train_batches, scaled_test_batches = [], []

        for batch in train_loader:
            n = batch.x.shape[0]
            batch.x = X_train_scaled[i_train:i_train + n]
            batch = batch.to(device)
            scaled_train_batches.append(batch)
            i_train += n

        for batch in test_loader:
            n = batch.x.shape[0]
            batch.x = X_test_scaled[i_test:i_test + n]
            batch = batch.to(device)
            scaled_test_batches.append(batch)
            i_test += n

        train_loader_scaled = torch.utils.data.DataLoader(scaled_train_batches, batch_size=None, shuffle=True)
        test_loader_scaled = torch.utils.data.DataLoader(scaled_test_batches, batch_size=None, shuffle=False)

    else:  # Regular tensor dataset
        train_scaled = torch.utils.data.TensorDataset(X_train_scaled, y_train)
        test_scaled = torch.utils.data.TensorDataset(X_test_scaled, y_test)

        train_loader_scaled = torch.utils.data.DataLoader(train_scaled, batch_size=64, shuffle=True)
        test_loader_scaled = torch.utils.data.DataLoader(test_scaled, batch_size=64, shuffle=False)

    return train_loader_scaled, test_loader_scaled, scaler


def preprocess_graph_list_inplace(graph_list, strategy='mean', device='cuda'):
    """
    In-place preprocess node features of a list of PyG graphs.
    Edge weights are left as-is.

    Parameters
    ----------
    graph_list : list of torch_geometric.data.Data
    strategy : str ('mean', 'median', 'zero')
    device : str or torch.device

    Returns
    -------
    x_scaler : fitted scaler for node features
    """
    # 1) Gather all node features into one big tensor
    x_all = torch.cat([g.x.to(device) for g in graph_list], dim=0)

    # 2) Preprocess node features (returns X_proc of same shape, plus scaler)
    X_proc, x_scaler = preprocess_features(x_all, strategy=strategy, device=device)

    # 3) Write back into each graph x, leave edge_weight alone
    offset = 0
    for g in graph_list:
        n_nodes = g.x.size(0)
        # overwrite g.x with the corresponding slice
        g.x = X_proc[offset : offset + n_nodes]
        offset += n_nodes

    return x_scaler