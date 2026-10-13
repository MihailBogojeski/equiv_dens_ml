from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

"""
Channel-wise Clebsch-Gordan tensor products as one sparse 4-linear form over the slots (x1, x2, w, y):

    S = sum_t cg_t * x1[n, i1_t, f] * x2[n, i2_t, f] * w[n, p_t, f] * y[n, io_t, f]

The product y = CGProduct(x1, x2, w) is dS/dy, and the gradient of any slot is dS/d(slot) with the output
gradient placed in the y slot. So forward, backward and double backward are all the same fused CUDA op
(`equiv_dens::cg_tp`): for every output row it loops over the triples (sorted by that slot) and accumulates
in registers, so no [batch, triples, F] intermediate is ever materialized.
"""


@triton.jit
def _cg_tp_kernel(a_ptr, b_ptr, c_ptr, out_ptr, rowptr_ptr, tab_ptr, coef_ptr, N, F,
                  sa, sb, sc, so, Da, Db, Dc,
                  COL_A: tl.constexpr, COL_B: tl.constexpr, COL_C: tl.constexpr,
                  A_BATCH: tl.constexpr, B_BATCH: tl.constexpr, C_BATCH: tl.constexpr,
                  REDUCE: tl.constexpr, ACC64: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_F: tl.constexpr):
    o = tl.program_id(0)
    f = tl.program_id(2) * BLOCK_F + tl.arange(0, BLOCK_F)
    fm = f < F
    start = tl.load(rowptr_ptr + o)
    end = tl.load(rowptr_ptr + o + 1)
    acc_t = tl.float64 if ACC64 else tl.float32
    if REDUCE:
        num_blocks = tl.cdiv(N, BLOCK_N)
        first = 0
    else:
        num_blocks = 1
        first = tl.program_id(1)
    total = tl.zeros([BLOCK_F], dtype=acc_t)
    for nb in range(first, first + num_blocks):
        n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (n < N)[:, None] & fm[None, :]
        acc = tl.zeros([BLOCK_N, BLOCK_F], dtype=acc_t)
        for t in range(start, end):
            ia = tl.load(tab_ptr + 4 * t + COL_A)
            ib = tl.load(tab_ptr + 4 * t + COL_B)
            ic = tl.load(tab_ptr + 4 * t + COL_C)
            cg = tl.load(coef_ptr + t).to(acc_t)
            if A_BATCH:
                a = tl.load(a_ptr + n[:, None] * sa + ia * F + f[None, :], mask=mask, other=0.).to(acc_t)
            else:
                a = tl.load(a_ptr + ia * F + f, mask=fm, other=0.).to(acc_t)[None, :]
            if B_BATCH:
                b = tl.load(b_ptr + n[:, None] * sb + ib * F + f[None, :], mask=mask, other=0.).to(acc_t)
            else:
                b = tl.load(b_ptr + ib * F + f, mask=fm, other=0.).to(acc_t)[None, :]
            if C_BATCH:
                c = tl.load(c_ptr + n[:, None] * sc + ic * F + f[None, :], mask=mask, other=0.).to(acc_t)
            else:
                c = tl.load(c_ptr + ic * F + f, mask=fm, other=0.).to(acc_t)[None, :]
            acc += cg * a * b * c
        if REDUCE:
            total += tl.sum(acc, axis=0)
        else:
            tl.store(out_ptr + n[:, None] * so + o * F + f[None, :], acc.to(out_ptr.dtype.element_ty), mask=mask)
    if REDUCE:
        tl.store(out_ptr + o * F + f, total.to(out_ptr.dtype.element_ty), mask=fm)


@triton_op("equiv_dens::cg_tp", mutates_args=())
def cg_tp(x1: Optional[torch.Tensor], x2: Optional[torch.Tensor], w: Optional[torch.Tensor],
          y: Optional[torch.Tensor], k: int, reduce: bool, dims: list[int], tab: torch.Tensor,
          coef: torch.Tensor, rowptr: torch.Tensor) -> torch.Tensor:
    """
    dS/d(slot k) for the slots (x1, x2, w, y) = (0, 1, 2, 3); the slot k argument is None.
    Slot tensors are [N or 1, D_slot, F] contiguous, a leading 1 broadcasts over N (shared weights).
    tab[k] / coef[k] / rowptr[k] are the triples sorted by slot k (columns i1, i2, p, io) and the CSR row pointer.
    With reduce the result is [1, dims[k], F] summed over N (the gradient of shared weights).
    """
    a, b, c = [x for x in (x1, x2, w, y) if x is not None]
    cols = [s for s in range(4) if s != k]
    N = max(a.shape[0], b.shape[0], c.shape[0])
    F = a.shape[-1]
    out = a.new_empty(1 if reduce else N, dims[k], F)
    BLOCK_F = min(128, triton.next_power_of_2(F))
    BLOCK_N = 16
    grid = (dims[k], 1 if reduce else triton.cdiv(N, BLOCK_N), triton.cdiv(F, BLOCK_F))
    wrap_triton(_cg_tp_kernel)[grid](
        a, b, c, out, rowptr[k], tab[k], coef[k], N, F,
        a.stride(0), b.stride(0), c.stride(0), out.stride(0), a.shape[1], b.shape[1], c.shape[1],
        COL_A=cols[0], COL_B=cols[1], COL_C=cols[2],
        A_BATCH=a.shape[0] != 1, B_BATCH=b.shape[0] != 1, C_BATCH=c.shape[0] != 1,
        REDUCE=reduce, ACC64=a.dtype == torch.float64, BLOCK_N=BLOCK_N, BLOCK_F=BLOCK_F,
    )
    return out


class CGTP(torch.autograd.Function):
    """cg_tp with its gradients: the gradient of slot j is cg_tp with the output gradient in slot k."""

    @staticmethod
    def forward(x1, x2, w, y, k, reduce, dims, tab, coef, rowptr):
        return cg_tp(x1, x2, w, y, k, reduce, dims, tab, coef, rowptr)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x1, x2, w, y, k, reduce, dims, tab, coef, rowptr = inputs
        ctx.save_for_backward(x1, x2, w, y)
        ctx.k, ctx.dims, ctx.tab, ctx.coef, ctx.rowptr = k, dims, tab, coef, rowptr

    @staticmethod
    def backward(ctx, grad):
        slots = list(ctx.saved_tensors)
        slots[ctx.k] = grad.contiguous()
        grads = [None] * 4
        for j in range(4):
            if j != ctx.k and ctx.needs_input_grad[j]:
                others = [None if s == j else slots[s] for s in range(4)]
                reduce = slots[j].shape[0] == 1 and grad.shape[0] != 1
                grads[j] = CGTP.apply(*others, j, reduce, ctx.dims, ctx.tab, ctx.coef, ctx.rowptr)
        return grads[0], grads[1], grads[2], grads[3], None, None, None, None, None, None


class CGProduct(nn.Module):
    """
    Channel-wise tensor product of two packed spherical features over a fixed list of (l1, l2, L) paths.

    out[..., M, f] = sum_paths w[..., path, f] * sum_{m1, m2} cg[m1, m2, M] * x1[..., m1, f] * x2[..., m2, f]

    The nonzero (m1, m2, M) Clebsch-Gordan triples of all paths are collected once. On the GPU the product is
    the fused `cg_tp` op, on the CPU a gather - multiply - scatter over the triples.
    """

    def __init__(self, cg_matrix, paths):
        super().__init__()
        self.paths = paths
        i1, i2, io, val, pid = [], [], [], [], []
        for p, (l1, l2, L) in enumerate(paths):
            block = cg_matrix[l1 ** 2:(l1 + 1) ** 2, l2 ** 2:(l2 + 1) ** 2, L ** 2:(L + 1) ** 2]
            a, b, c = torch.nonzero(block, as_tuple=True)
            i1.append(a + l1 ** 2)
            i2.append(b + l2 ** 2)
            io.append(c + L ** 2)
            val.append(block[a, b, c])
            pid.append(torch.full_like(a, p))
        self.dims = [cg_matrix.shape[0], cg_matrix.shape[1], len(paths), cg_matrix.shape[2]]
        idx = torch.stack([torch.cat(i1), torch.cat(i2), torch.cat(pid), torch.cat(io)], dim=1)
        val = torch.cat(val)
        for name, v in [('i1', idx[:, 0]), ('i2', idx[:, 1]), ('pid', idx[:, 2]), ('io', idx[:, 3])]:
            self.register_buffer(name, v.clone(), persistent=False)
        self.register_buffer('cg', val, persistent=False)
        # triples sorted by every slot, with CSR row pointers, for the fused op
        tab, coef, rowptr = [], [], []
        for s in range(4):
            order = torch.sort(idx[:, s], stable=True).indices
            tab.append(idx[order])
            coef.append(val[order])
            counts = torch.bincount(idx[:, s], minlength=self.dims[s])
            rowptr.append(nn.functional.pad(torch.cumsum(counts, 0), (1, max(self.dims) - self.dims[s])))
        self.register_buffer('tab', torch.stack(tab).int().contiguous(), persistent=False)
        self.register_buffer('coef', torch.stack(coef).contiguous(), persistent=False)
        self.register_buffer('rowptr', torch.stack(rowptr).int().contiguous(), persistent=False)

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
        D_out = self.dims[3]
        x1 = x1.reshape(-1, x1.shape[-2], F_)
        x2 = x2.reshape(-1, x2.shape[-2], F_)
        w = w.reshape(-1, w.shape[-2], F_)
        if x1.is_cuda:
            y = CGTP.apply(x1.contiguous(), x2.contiguous(), w.contiguous(), None, 3, False, self.dims,
                      self.tab, self.coef.to(x1.dtype), self.rowptr)
            return y.view(*lead, D_out, F_)
        prod = x1[:, self.i1] * x2[:, self.i2] * self.cg.to(x1.dtype).view(1, -1, 1) * w[:, self.pid]
        y = x1.new_zeros(x1.shape[0], D_out, F_).index_add(1, self.io, prod)
        return y.view(*lead, D_out, F_)
