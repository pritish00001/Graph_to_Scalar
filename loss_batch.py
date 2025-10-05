# losses_batch.py
import torch
import torch.nn.functional as F
from typing import Iterable, List, Optional, Tuple, Union


def _batch_to_graph_iter(batch):
    """
    Generic iterator that yields per-graph tuples (x, U, y, graph_idx).
    Supports:
      - PyG Batch with concatenated tensors: batch.x, batch.U (optional), batch.ptr or batch.batch
      - PyG Batch where each graph stores its own U: batch.to_data_list() (Data objects)
    Yields:
      (x: Tensor [Ni, F], U: Tensor [Ni, M] or None, y: Tensor or None, idx: int)
    """
    # 1) If the batch has to_data_list(), prefer that (gives per-graph Data objects)
    if hasattr(batch, "to_data_list"):
        data_list = batch.to_data_list()
        for i, data in enumerate(data_list):
            x = getattr(data, "x", None)
            U = getattr(data, "U", None) or getattr(data, "eigvecs", None) or getattr(data, "eigenvectors", None)
            y = getattr(data, "y", None)
            yield x, U, y, i
        return

    # 2) Otherwise, try concatenated tensors + ptr / batch pointer
    # Prefer ptr: cumulative node counts, shape [B+1]
    if hasattr(batch, "ptr"):
        ptr = batch.ptr
        total_nodes = ptr[-1].item()
        x_all = getattr(batch, "x", None)
        U_all = getattr(batch, "U", None) or getattr(batch, "eigvecs", None) or getattr(batch, "eigenvectors", None)
        y_all = getattr(batch, "y", None)
        B = ptr.size(0) - 1
        for i in range(B):
            s = int(ptr[i].item())
            e = int(ptr[i + 1].item())
            x = x_all[s:e] if x_all is not None else None
            U = U_all[s:e] if U_all is not None else None
            # y may be per-graph (length B) or per-node (length total_nodes)
            if y_all is None:
                y = None
            else:
                if y_all.dim() == 1 and y_all.size(0) == B:
                    # assume per-graph labels
                    y = y_all[i]
                elif y_all.size(0) == total_nodes:
                    y = y_all[s:e]
                else:
                    # unknown shape: return the whole and let caller decide
                    y = y_all
            yield x, U, y, i
        return

    # 3) Fallback: if batch is a tuple (X_batch, U_batch, y_batch, ...) with tensors
    if isinstance(batch, (tuple, list)):
        # assume shapes: (X_all, U_all, y_all, ptr) or (X_all, U_all, y_all, batch_index)
        # Try common conventions:
        # - (X_all, y_graph) per-batch
        # We will try to be conservative and yield whole tensors as single graph
        X_all = batch[0] if len(batch) > 0 else None
        U_all = batch[1] if len(batch) > 1 else None
        y_all = batch[2] if len(batch) > 2 else None
        # Treat as a single graph bundle
        yield X_all, U_all, y_all, 0
        return

    # 4) If nothing matched, raise error
    raise ValueError("Unsupported batch type in _batch_to_graph_iter. "
                     "Batch must be a PyG Batch with .to_data_list() or .ptr/.x, or a (tuple/list) bundle.")


def _select_eigvecs(U: torch.Tensor, K1: int, K2: int) -> torch.Tensor:
    """
    Select first K1 and last K2 columns from U (shape [N, M]) -> [N, K1+K2].
    """
    if U is None:
        raise ValueError("U is None — cannot select eigenvectors.")
    N, M = U.shape
    K = K1 + K2
    if K == 0:
        # return empty tensor with shape [N, 0]
        return U.new_empty((N, 0))
    if K1 > M or K2 > M:
        raise ValueError(f"K1 or K2 exceed available eigenvector columns (M={M}).")
    if K2 == 0:
        return U[:, :K1].contiguous()
    if K1 == 0:
        return U[:, -K2:].contiguous()
    return torch.cat([U[:, :K1], U[:, -K2:]], dim=1).contiguous()


class LossesBatch:
    @staticmethod
    def eigen_alignment_loss_from_batches(
        batch_real,
        batch_syn,
        K1_list: Optional[Union[List[int], int]] = None,
        K2_list: Optional[Union[List[int], int]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute eigen-alignment loss given two batches (real and synthetic).
        Parameters:
          - batch_real, batch_syn: batches from DataLoader (PyG Batch recommended).
            Each graph must provide x (node features) and U (eigenvectors) either as:
              * per-graph attributes via batch.to_data_list() where data.U exists, OR
              * concatenated: batch.x [sumNi, F] and batch.U [sumNi, M] and batch.ptr available.
          - K1_list, K2_list: either lists of ints length=B, or scalars applied to all graphs,
            or None (if graphs carry attributes 'K1'/'K2' in their data objects).
          - device: optional device to run on (defaults to device of input tensors).
        Returns:
          scalar tensor (mean loss across graphs in the batch)
        """

        # Build per-graph iterators
        iter_real = list(_batch_to_graph_iter(batch_real))
        iter_syn = list(_batch_to_graph_iter(batch_syn))

        if len(iter_real) != len(iter_syn):
            raise ValueError("Real and synthetic batches must contain the same number of graphs in the batch.")

        B = len(iter_real)

        # Normalize K1_list / K2_list into lists
        if K1_list is None:
            # try to get per-graph K1 attribute if available
            # if batches were to_data_list, check for K1 in data; otherwise default to 0
            K1_list_use = []
            for (xr, Ur, yr, idx), (xs, Us, ys, idxs) in zip(iter_real, iter_syn):
                k1 = 0
                # check for attribute 'K1' in syn or real data: not available here unless to_data_list used
                K1_list_use.append(k1)
        elif isinstance(K1_list, int):
            K1_list_use = [K1_list] * B
        else:
            K1_list_use = list(K1_list)

        if K2_list is None:
            K2_list_use = [0] * B
        elif isinstance(K2_list, int):
            K2_list_use = [K2_list] * B
        else:
            K2_list_use = list(K2_list)

        losses = []
        for i, ((xr, Ur, yr, _), (xs, Us, ys, _)) in enumerate(zip(iter_real, iter_syn)):
            # infer device
            if device is None:
                dev = xr.device if xr is not None else (xs.device if xs is not None else torch.device("cpu"))
            else:
                dev = device

            # move to device and ensure not None
            if xr is None or xs is None:
                raise ValueError(f"Missing x (node features) for graph {i} in one of the batches.")
            xr = xr.to(dev)
            xs = xs.to(dev)

            if Ur is None:
                raise ValueError(f"Missing U (eigenvectors) for graph {i} in real batch.")
            if Us is None:
                raise ValueError(f"Missing U (eigenvectors) for graph {i} in synthetic batch.")
            Ur = Ur.to(dev)
            Us = Us.to(dev)

            K1 = K1_list_use[i]
            K2 = K2_list_use[i]
            K = K1 + K2
            if K == 0:
                # if K==0, P becomes zero matrix; alignment is trivial (loss zero)
                losses.append(xr.new_tensor(0.0))
                continue

            # Validate dims
            Ni, F = xr.shape
            Nis, Fs = xs.shape
            if F != Fs:
                raise ValueError(f"Feature dim mismatch: real F={F}, syn F={Fs} at graph {i}")
            if Ur.shape[0] != Ni:
                raise ValueError(f"U_real rows {Ur.shape[0]} != nodes {Ni} for graph {i}")
            if Us.shape[0] != Nis:
                raise ValueError(f"U_syn rows {Us.shape[0]} != nodes {Nis} for graph {i}")

            # select eigenvectors
            Ur_sel = _select_eigvecs(Ur, K1, K2)  # [Ni, K]
            Us_sel = _select_eigvecs(Us, K1, K2)  # [Nis, K]
            if Ur_sel.shape[1] != Us_sel.shape[1]:
                raise ValueError(f"K mismatch between real and syn for graph {i}")

            # V = U^T X  -> [K, F]
            V_r = Ur_sel.t() @ xr       # [K, F]
            V_s = Us_sel.t() @ xs       # [K, F]

            # P = V^T V  -> [F, F]
            P_r = V_r.t() @ V_r
            P_s = V_s.t() @ V_s

            loss_i = torch.norm(P_r - P_s, p='fro') ** 2
            losses.append(loss_i)

        losses = torch.stack(losses, dim=0) if len(losses) > 0 else torch.tensor(0.0, device=device or torch.device("cpu"))
        return losses.mean()

    @staticmethod
    def orthogonality_loss_from_batch(
        batch_syn,
        K_select: Optional[Union[List[int], int]] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Orthogonality loss computed for synthetic batch. Uses batch.U or per-data U.
        K_select: if provided, selects first K/K2 columns similarly to eigen selection;
                  can be an int for all graphs or list per-graph. If None, uses all columns.
        """
        iter_syn = list(_batch_to_graph_iter(batch_syn))
        losses = []
        for i, (xs, Us, ys, _) in enumerate(iter_syn):
            if Us is None:
                raise ValueError(f"Missing U in synthetic batch for graph {i}.")
            dev = device or Us.device
            Us = Us.to(dev)

            # decide K for this graph
            if K_select is None:
                K = Us.shape[1]
                Us_sel = Us
            else:
                if isinstance(K_select, int):
                    k1 = K_select
                    k2 = 0
                else:
                    # if list, assume entry is K (total) or tuple (K1,K2)
                    entry = K_select[i]
                    if isinstance(entry, tuple):
                        k1, k2 = entry
                    else:
                        k1, k2 = int(entry), 0
                Us_sel = _select_eigvecs(Us, k1, k2)

            if Us_sel.shape[1] == 0:
                losses.append(torch.tensor(0.0, device=dev))
                continue

            UtU = Us_sel.t() @ Us_sel  # [K, K]
            I = torch.eye(UtU.shape[0], device=dev, dtype=UtU.dtype)
            loss_i = torch.norm(UtU - I, p='fro') ** 2
            losses.append(loss_i)

        losses = torch.stack(losses, dim=0) if len(losses) > 0 else torch.tensor(0.0)
        return losses.mean()

    @staticmethod
    def regression_loss_from_batch(
        y_pred: torch.Tensor,
        batch_true,
        device: Optional[torch.device] = None,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute regression loss comparing y_pred (output of model, per-graph or per-batch)
        to true labels in batch_true (which can be batch with per-graph y or per-node y).
        y_pred shape must match y_true after extraction.
        """
        # extract y_true: if batch has per-graph y of length B, use that
        # try batch_true.to_data_list() first
        ys = []
        if hasattr(batch_true, "to_data_list"):
            for data in batch_true.to_data_list():
                y = getattr(data, "y", None)
                if y is None:
                    raise ValueError("Each Data in batch_true must contain attribute 'y' for regression.")
                ys.append(y)
            # stack if scalar per-graph
            if ys and ys[0].dim() == 0:
                y_true = torch.stack([y.view(()) for y in ys], dim=0)
            else:
                # if y are vectors, pad/stack (user must ensure shapes)
                y_true = torch.stack(ys, dim=0)
        elif hasattr(batch_true, "y"):
            y_all = batch_true.y
            # if y_all length matches number of graphs (B), use directly
            if y_all.dim() == 1 and hasattr(batch_true, "ptr") and y_all.shape[0] == batch_true.ptr.numel() - 1:
                y_true = y_all.to(device or y_all.device)
            else:
                # fallback: if y_pred is per-node, ensure shapes match
                y_true = y_all.to(device or y_all.device)
        else:
            raise ValueError("Could not find labels in batch_true (no .y or .to_data_list with .y).")

        if device is not None:
            y_true = y_true.to(device)

        return F.mse_loss(y_pred, y_true, reduction=reduction)

    @staticmethod
    def synthetic_graph_loss(eig_loss, ortho_loss, reg_loss, alpha=1.0, beta=1.0, gamma=1.0):
        return alpha * eig_loss + beta * ortho_loss + gamma * reg_loss
