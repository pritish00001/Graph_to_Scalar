from load_data import load_train_test
from preprocess import scale_dataloaders, collect_all_batches
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)

train_loader, test_loader = load_train_test()

train_loader_scaled, test_loader_scaled, scaler = scale_dataloaders(train_loader, test_loader, device=device)
X,y = collect_all_batches(train_loader)
print(X[0])
X,y = collect_all_batches(train_loader_scaled)
print(X[0])
