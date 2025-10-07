import torch
import torch.nn.functional as F

class Losses:
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
