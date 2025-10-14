import torch
import torch.nn.functional as F

class Losses:
    
    @staticmethod
    def eigen_alignment_loss_concatenated(X_real, U_real, X_syn, U_syn, batch_real, batch_syn):
        """
        Eigen alignment loss for concatenated graph batches.
        
        Args:
            X_real: [N_sum_real, F] node features of all real graphs concatenated
            U_real: [N_sum_real, K_max] eigenvectors of all real graphs concatenated
            X_syn:  [N_sum_syn, F] node features of all synthetic graphs concatenated
            U_syn:  [N_sum_syn, K_max] eigenvectors of all synthetic graphs concatenated
            batch_real: [N_sum_real] graph assignment indices for real graphs
            batch_syn: [N_sum_syn] graph assignment indices for synthetic graphs
            
        Returns:
            Scalar loss (mean over graphs)
        """
        B_real = batch_real.max().item() + 1
        B_syn = batch_syn.max().item() + 1
        assert B_real == B_syn, "Number of real and synthetic graphs must match"

        loss_per_graph = []

        for b in range(B_real):
            # Mask nodes belonging to graph b
            mask_r = batch_real == b
            mask_s = batch_syn == b

            Xr = X_real[mask_r]  # [N_b_real, F]
            Ur = U_real[mask_r]  # [N_b_real, K_b]

            Xs = X_syn[mask_s]   # [N_b_syn, F]
            Us = U_syn[mask_s]   # [N_b_syn, K_b]

            # Compute Pr and Ps
            Pr = Xr.T @ Ur @ Ur.T @ Xr  # [F, F]
            Ps = Xs.T @ Us @ Us.T @ Xs  # [F, F]

            loss_per_graph.append(torch.norm(Pr - Ps, p='fro')**2)

        loss = torch.stack(loss_per_graph).mean()
        return loss

    @staticmethod
    # def eigen_alignment_loss(X_real, U_real, X_syn, U_syn, K1_list, K2_list):
    #     """
    #     Fully vectorized eigen alignment loss across a batch of graphs.
    #     Pads nodes and K dimension to max values in batch.
    #     """
    #     B = len(X_real)
    #     N_max = max(x.shape[0] for x in X_real + X_syn)
    #     F_dim = X_real[0].shape[1]
    #     K_max = max([K1 + K2 for K1, K2 in zip(K1_list, K2_list)])

    #     # Pad X_real, X_syn
    #     Xr_batch = torch.stack([F.pad(x, (0,0,0,N_max - x.shape[0])) for x in X_real])  # [B, N_max, F]
    #     Xs_batch = torch.stack([F.pad(x, (0,0,0,N_max - x.shape[0])) for x in X_syn])

    #     # Pad U_real_sel, U_syn_sel
    #     Ur_batch = torch.stack([F.pad(torch.cat([U[:, :K1], U[:, -K2:]], dim=1),
    #                                  (0, K_max - (K1+K2), 0, N_max - U.shape[0]))
    #                             for U, K1, K2 in zip(U_real, K1_list, K2_list)])
    #     Us_batch = torch.stack([F.pad(U[:, :Ur_batch.shape[1]],
    #                                  (0, K_max - U[:, :Ur_batch.shape[1]].shape[1], 0, N_max - U.shape[0]))
    #                             for U in U_syn])

    #     # Compute Pr and Ps using einsum: batch matrix multiplication
    #     # Pr[b] = Xr[b].T @ Ur[b] @ Ur[b].T @ Xr[b]
    #     Pr = torch.einsum('bnf,bnk,bnk,bnm->bfm', Xr_batch, Ur_batch, Ur_batch, Xr_batch)
    #     Ps = torch.einsum('bnf,bnk,bnk,bnm->bfm', Xs_batch, Us_batch, Us_batch, Xs_batch)

    #     # Frobenius norm per graph
    #     loss = torch.norm(Pr - Ps, dim=(1,2))**2
    #     return loss.mean()

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
