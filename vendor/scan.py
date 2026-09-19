"""A selective scan that needs no CUDA kernel, so MambaIRv2 can run anywhere.

Vendored from the Inkvec repository, ``tools/inkvec_sr/scan.py`` (LogoLabs, Apache-2.0).
The scan itself (``_scan_chunk`` and the body of ``parallel_scan``) is unchanged. What
changed for ComfyUI: the reference ``install()`` bound a stand-in ``mamba_ssm`` package into
``sys.modules``. In a ComfyUI process that would be visible to every other custom node, so
here the architecture file imports ``selective_scan_fn`` from this module instead, and this
module dispatches per call: mamba-ssm's CUDA kernel when the tensors are on CUDA and
``mamba_ssm`` imports cleanly, the pure-PyTorch scan otherwise. Nothing is registered in
``sys.modules`` beyond what a normal ``import mamba_ssm`` does.

From the reference docstring: `mamba-ssm` ships its scan as a compiled CUDA extension and
publishes no Windows wheel. The scan is one first-order linear recurrence per
(batch, channel, state) triple:

    x_i = a_i * x_{i-1} + b_i,        y_i = <x_i, C_i> + D * u_i

which is associative -- the pair (a, b) composes as

    (a1, b1) . (a2, b2) = (a1 a2,  a2 b1 + b2)

-- and therefore parallel. A Hillis-Steele doubling pass computes the whole prefix in
log2(L) steps instead of L. The a-products are exp(negative) and stay inside (0, 1], so the
formulation is numerically safe. The recurrence runs in chunks along the sequence; only the
carried state crosses a chunk boundary, so the result is exact rather than approximate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Sequence chunk. Larger is fewer Python steps and more peak memory.
CHUNK = 256

_cuda_kernel = None  # None: not looked up yet; False: unavailable; else the function


def _scan_chunk(a: torch.Tensor, b: torch.Tensor,
                carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive scan of x_i = a_i x_{i-1} + b_i along dim -2, starting at `carry`.

    a, b: (..., L, N).  carry: (..., N).  Returns (x, last_state).
    """
    b = b.clone()
    b[..., 0, :] = b[..., 0, :] + a[..., 0, :] * carry
    a = a.clone()
    L = a.shape[-2]
    d = 1
    while d < L:
        # Each element currently holds the recurrence run over the last d steps;
        # composing it with the one d places back doubles that reach.
        a_lo, b_lo = a[..., :-d, :], b[..., :-d, :]
        a_hi = a[..., d:, :]
        b[..., d:, :] = b[..., d:, :] + a_hi * b_lo
        a[..., d:, :] = a_hi * a_lo
        d *= 2
    return b, b[..., -1, :]


def parallel_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                  delta_softplus=False, return_last_state=False):
    """Pure-PyTorch drop-in for `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`.

    Covers the shapes MambaIRv2 uses: u, delta (b, d, L); A (d, N);
    B, C (b, g, N, L) with d a multiple of g; D (d,); delta_bias (d,).
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    bsz, dim, L = u.shape
    N = A.shape[1]

    def expand(t: torch.Tensor) -> torch.Tensor:
        # (b, g, N, L) -> (b, d, L, N), each group repeated d/g times, matching
        # the flattening `xs.view(b, -1, L)` of an original (b, g, d/g, L).
        g = t.shape[1]
        t = t.float().unsqueeze(2).expand(bsz, g, dim // g, N, L)
        return t.reshape(bsz, dim, N, L).permute(0, 1, 3, 2)

    Bx = expand(B)
    Cx = expand(C)
    A = A.float()

    x = A.new_zeros((bsz, dim, N))
    ys = []
    for s in range(0, L, CHUNK):
        e = min(s + CHUNK, L)
        d_ = delta[:, :, s:e]                                        # (b, d, c)
        a = torch.exp(d_.unsqueeze(-1) * A[None, :, None, :])        # (b, d, c, N)
        b = d_.unsqueeze(-1) * Bx[:, :, s:e] * u[:, :, s:e].unsqueeze(-1)
        xs, x = _scan_chunk(a, b, x)
        ys.append((xs * Cx[:, :, s:e]).sum(-1))                      # (b, d, c)

    y = torch.cat(ys, dim=-1)
    if D is not None:
        y = y + u * D.float()[:, None]
    if z is not None:
        y = y * F.silu(z.float())
    out = y.to(dtype_in)
    return (out, x) if return_last_state else out


def _kernel():
    """mamba-ssm's CUDA selective scan, or None when it cannot be imported."""
    global _cuda_kernel
    if _cuda_kernel is None:
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as fn

            _cuda_kernel = fn
        except Exception:  # not installed, or built against another torch/CUDA
            _cuda_kernel = False
    return _cuda_kernel or None


def backend(device) -> str:
    """Which scan runs on `device`, for logging."""
    if str(device).startswith("cuda") and _kernel() is not None:
        return "mamba-ssm CUDA kernel"
    return "pure-PyTorch parallel scan"


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """Dispatch: the CUDA kernel for CUDA tensors when available, else the parallel scan."""
    if u.is_cuda:
        fn = _kernel()
        if fn is not None:
            return fn(u, delta, A, B, C, D, z=z, delta_bias=delta_bias,
                      delta_softplus=delta_softplus, return_last_state=return_last_state)
    return parallel_scan(u, delta, A, B, C, D, z=z, delta_bias=delta_bias,
                         delta_softplus=delta_softplus, return_last_state=return_last_state)


selective_scan_ref = selective_scan_fn
