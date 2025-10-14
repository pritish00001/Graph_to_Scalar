import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

class Losses:
    
    @staticmethod
    def eigen_alignment_loss_concatenated(
        X_real, U_real, X_syn, U_syn, batch_real, batch_syn
    ):
        """
        Eigen alignment loss for concatenated graph batches (no manual for-loop).

        Args:
            X_real: [N_sum_real, F]  node features (real graphs concatenated)
            U_real: [N_sum_real, K]  eigenvectors (real)
            X_syn:  [N_sum_syn, F]   node features (synthetic)
            U_syn:  [N_sum_syn, K]   eigenvectors (synthetic)
            batch_real: [N_sum_real] graph assignment indices (0 … B-1)
            batch_syn:  [N_sum_syn]  graph assignment indices (0 … B-1)

        Returns:
            Scalar loss (mean over graphs)
        """
        B_real = int(batch_real.max().item()) + 1
        B_syn = int(batch_syn.max().item()) + 1
        assert B_real == B_syn, "Mismatch between real and synthetic batch counts"
        B = B_real

        # Compute local covariance-like structures
        # Step 1: P_r_i = X_i.T U_i U_i.T X_i  ⇒ We rewrite as (X_i^T U_i)(U_i^T X_i)
        # Instead of looping, we compute grouped sums using scatter_add.

        # First compute XU = X.T @ U per graph
        # => elementwise product per node, then scatter sum per batch
        XU_real = X_real.unsqueeze(2) * U_real.unsqueeze(1)  # [N_sum_real, F, K]
        XU_syn  = X_syn.unsqueeze(2)  * U_syn.unsqueeze(1)   # [N_sum_syn, F, K]

        # Aggregate by batch (sum over nodes belonging to same graph)
        XU_r = scatter_add(XU_real, batch_real, dim=0, dim_size=B)  # [B, F, K]
        XU_s = scatter_add(XU_syn,  batch_syn,  dim=0, dim_size=B)  # [B, F, K]

        # Compute P_r = (XU_r)(XU_r^T) = XU_r @ XU_r^T per batch
        Pr = torch.matmul(XU_r, XU_r.transpose(1, 2))  # [B, F, F]
        Ps = torch.matmul(XU_s, XU_s.transpose(1, 2))  # [B, F, F]

        # Frobenius loss per graph
        loss_per_graph = torch.norm(Pr - Ps, dim=(1, 2)) ** 2  # [B]

        return loss_per_graph.mean()

    @staticmethod
    def orthogonality_loss_concatenated(U_syn, batch_syn):
        """
        Orthogonality loss for concatenated eigenvector batches (no manual for-loop).
        
        Args:
            U_syn: [N_sum_syn, K]  concatenated eigenvectors of synthetic graphs
            batch_syn: [N_sum_syn] graph assignment indices (0 … B-1)
        
        Returns:
            Scalar loss (mean over graphs)
        """

        B = int(batch_syn.max().item()) + 1
        K = U_syn.shape[1]

        # Step 1: compute U^T U per graph
        # Equivalent to sum_i (U_i^T U_i) over nodes in each graph
        # For each graph, we need U_batch^T U_batch = (sum over n of U[n,k1]*U[n,k2])
        # That’s a grouped outer product sum over nodes.

        # Compute pairwise products U[n,k1]*U[n,k2] per node → [N_sum, K, K]
        U_outer = torch.einsum('nk,nl->nkl', U_syn, U_syn)  # [N_sum_syn, K, K]

        # Sum within each graph
        UU = scatter_add(U_outer, batch_syn, dim=0, dim_size=B)  # [B, K, K]

        # Step 2: orthogonality deviation per graph
        I = torch.eye(K, device=U_syn.device).unsqueeze(0)  # [1, K, K]
        ortho_diff = UU - I  # [B, K, K]

        # Step 3: Frobenius loss per graph
        loss_per_graph = torch.norm(ortho_diff, dim=(1, 2)) ** 2  # [B]
        return loss_per_graph.mean()

    @staticmethod
    def eigen_alignment_loss(X_real, U_real, X_syn, U_syn, K1_list, K2_list):
        """
        Fully vectorized eigen alignment loss across a batch of graphs.
        Pads nodes and K dimension to max values in batch.
        """
        B = len(X_real)
        N_max = max(x.shape[0] for x in X_real + X_syn)
        F_dim = X_real[0].shape[1]
        K_max = max([K1 + K2 for K1, K2 in zip(K1_list, K2_list)])

        # Pad X_real, X_syn
        Xr_batch = torch.stack([F.pad(x, (0,0,0,N_max - x.shape[0])) for x in X_real])  # [B, N_max, F]
        Xs_batch = torch.stack([F.pad(x, (0,0,0,N_max - x.shape[0])) for x in X_syn])

        # Pad U_real_sel, U_syn_sel
        Ur_batch = torch.stack([F.pad(torch.cat([U[:, :K1], U[:, -K2:]], dim=1),
                                     (0, K_max - (K1+K2), 0, N_max - U.shape[0]))
                                for U, K1, K2 in zip(U_real, K1_list, K2_list)])
        Us_batch = torch.stack([F.pad(U[:, :Ur_batch.shape[1]],
                                     (0, K_max - U[:, :Ur_batch.shape[1]].shape[1], 0, N_max - U.shape[0]))
                                for U in U_syn])

        # Compute Pr and Ps using einsum: batch matrix multiplication
        # Pr[b] = Xr[b].T @ Ur[b] @ Ur[b].T @ Xr[b]
        Pr = torch.einsum('bnf,bnk,bnk,bnm->bfm', Xr_batch, Ur_batch, Ur_batch, Xr_batch)
        Ps = torch.einsum('bnf,bnk,bnk,bnm->bfm', Xs_batch, Us_batch, Us_batch, Xs_batch)

        # Frobenius norm per graph
        loss = torch.norm(Pr - Ps, dim=(1,2))**2
        return loss.mean()

    def eigen_alignment_loss_batched(X_real_batch, U_real_batch, X_syn_batch, U_syn_batch):
        """
        Eigen alignment loss for batched graphs (already padded / batched).
        
        Args:
            X_real_batch: [B, N_max, F] real node features batch
            U_real_batch: [B, N_max, K_max] real eigenvectors batch
            X_syn_batch:  [B, N_max, F] synthetic node features batch
            U_syn_batch:  [B, N_max, K_max] synthetic eigenvectors batch
            
        Returns:
            Scalar loss (mean over batch)
        """
        # Compute Pr and Ps using einsum (batch matrix multiplication)
        # Pr[b] = Xr[b].T @ Ur[b] @ Ur[b].T @ Xr[b]
        Pr = torch.einsum('bnf,bnk,bnk,bnm->bfm', X_real_batch, U_real_batch, U_real_batch, X_real_batch)
        Ps = torch.einsum('bnf,bnk,bnk,bnm->bfm', X_syn_batch, U_syn_batch, U_syn_batch, X_syn_batch)

        # Frobenius norm per graph
        loss = torch.norm(Pr - Ps, dim=(1, 2)) ** 2

        return loss.mean()

    @staticmethod
    def orthogonality_loss(U_syn_list):
        """
        Fully vectorized orthogonality loss across a batch of graphs.
        """
        B = len(U_syn_list)
        N_max = max(U.shape[0] for U in U_syn_list)
        K_max = max(U.shape[1] for U in U_syn_list)

        # Pad Us
        Us_batch = torch.stack([F.pad(U, (0, K_max - U.shape[1], 0, N_max - U.shape[0])) for U in U_syn_list])
        I = torch.eye(K_max, device=Us_batch.device).unsqueeze(0)  # [1, K, K]

        # Compute Us^T @ Us - I for each graph
        ortho = torch.bmm(Us_batch.transpose(1,2), Us_batch) - I  # [B, K, K]
        loss = torch.norm(ortho, dim=(1,2))**2
        return loss.mean()

    @staticmethod
    def regression_loss(y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    @staticmethod
    def synthetic_graph_loss(eig_loss, ortho_loss, reg_loss, alpha, beta, gamma):
        return alpha * eig_loss + beta * ortho_loss + gamma * reg_loss
