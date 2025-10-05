from enriched_data import enrich_save_and_get_dataloader
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

out_dir = "enriched_graphs"
enriched_loader, file_paths = enrich_save_and_get_dataloader(
    train_loader_scaled,
    out_dir,
    out_batch_size=16,
    out_shuffle=False,
    compute_eig=True,
    batch_eig_max_N=200,
    device=device,
    adj_threshold=1e-6,
    save_prefix="enriched",
    max_graphs=None,
    num_workers_out=0,
    show_progress=True
)

# Use enriched_loader like any PyG DataLoader:
for batch in enriched_loader:
    # batch is a PyG Batch created by collating saved Data objects
    print(batch.x.shape, getattr(batch, "lap_eigvecs", None) is not None)
    break