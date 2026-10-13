import torch
import torch.nn as nn


class CGProduct(nn.Module):
    """
    Channel-wise tensor product of two packed spherical features over a fixed list of (l1, l2, L) paths.

    out[..., M, f] = sum_paths w[..., path, f] * sum_{m1, m2} cg[m1, m2, M] * x1[..., m1, f] * x2[..., m2, f]

    The nonzero (m1, m2, M) Clebsch-Gordan triples of all paths are collected once, so the product is a
    gather - multiply - scatter over the triples instead of a python loop over the paths.
    """

    def __init__(self, cg_matrix, paths):
        super().__init__()
        i1, i2, io, val, pid, kid = [], [], [], [], [], []
        key_io, key_pid = [], []
        for p, (l1, l2, L) in enumerate(paths):
            block = cg_matrix[l1 ** 2:(l1 + 1) ** 2, l2 ** 2:(l2 + 1) ** 2, L ** 2:(L + 1) ** 2]
            a, b, c = torch.nonzero(block, as_tuple=True)
            i1.append(a + l1 ** 2)
            i2.append(b + l2 ** 2)
            io.append(c + L ** 2)
            val.append(block[a, b, c])
            pid.append(torch.full_like(a, p))
            # triples with the same (path, output m) share the path weight
            kid.append(c + len(key_io))
            key_io.extend(range(L ** 2, (L + 1) ** 2))
            key_pid.extend([p] * (2 * L + 1))
        self.dim_out = cg_matrix.shape[2]
        for name, v in [('i1', i1), ('i2', i2), ('io', io), ('pid', pid), ('kid', kid)]:
            self.register_buffer(name, torch.cat(v), persistent=False)
        self.register_buffer('cg', torch.cat(val), persistent=False)
        self.register_buffer('key_io', torch.tensor(key_io), persistent=False)
        self.register_buffer('key_pid', torch.tensor(key_pid), persistent=False)

    def forward(self, x1, x2, w):
        """
        Args:
            x1: [..., D1, F]
            x2: [..., D2, F]
            w: per path weights, [num_paths, F] (shared) or [..., num_paths, F] (one per leading index)
        Returns:
            y: [..., D_out, F]
        """
        lead = x1.shape[:-2]
        F_ = x1.shape[-1]
        x1 = x1.reshape(-1, x1.shape[-2], F_)
        x2 = x2.reshape(-1, x2.shape[-2], F_)
        cg = self.cg.to(x1.dtype).view(1, -1, 1)
        prod = x1[:, self.i1] * x2[:, self.i2] * cg
        if w.dim() == 2:
            prod = prod * w[self.pid]
            y = x1.new_zeros(x1.shape[0], self.dim_out, F_).index_add(1, self.io, prod)
        else:
            w = w.reshape(-1, w.shape[-2], F_)
            u = x1.new_zeros(x1.shape[0], self.key_io.shape[0], F_).index_add(1, self.kid, prod)
            y = x1.new_zeros(x1.shape[0], self.dim_out, F_).index_add(1, self.key_io, u * w[:, self.key_pid])
        return y.view(*lead, self.dim_out, F_)
