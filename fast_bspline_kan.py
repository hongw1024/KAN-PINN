"""
Fast B-Spline KAN (PINN-optimised version).

Compared with the original fast_bspline_kan.py, this version fixes the three
main accuracy gaps versus the official pykan implementation while keeping the
same speed advantage:

  1. Spline coefficient initialisation is NO LONGER zero.
     Zero-init kills early gradient flow through the spline branch — the layer
     behaves like a tiny SiLU MLP for the first few hundred epochs, which is
     particularly damaging for PINNs that rely on second derivatives.
     We now use pykan-style noise initialisation:
         coef ~ U(-sigma/sqrt(n_basis), +sigma/sqrt(n_basis))

  2. Per-edge spline scale (scale_sp) and per-edge residual scale (scale_base)
     replace the original per-output `scale` + per-edge `w_b` pair.
     This matches pykan and significantly increases expressivity at virtually
     no extra cost (out*(in-1) extra parameters per layer).

  3. The `.item()` calls inside `forward` are removed to avoid CPU↔GPU sync.

Drop-in replacement — public API is unchanged:

    from fast_bspline_kan import KAN
    model = KAN(width=[2, 10, 10, 3], grid=25, k=5, device=device).to(torch.float64)
    y = model(x)   # x : (B, width[0])
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable


# ── Knot-vector helper ───────────────────────────────────────────────────────

def _extend_grid(grid: torch.Tensor, k: int) -> torch.Tensor:
    """Extend a uniform grid with k clamped knots on each side."""
    h     = grid[1] - grid[0]
    left  = grid[0]  - h * torch.arange(k, 0, -1, device=grid.device, dtype=grid.dtype)
    right = grid[-1] + h * torch.arange(1, k + 1, device=grid.device, dtype=grid.dtype)
    return torch.cat([left, grid, right])


# ── B-spline basis (vectorised de Boor) ─────────────────────────────────────

def bspline_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Evaluate B-spline basis of order k.

    x    : (B, d)         inputs pre-clamped to [grid[k], grid[-(k+1)]]
    grid : (G+1+2k,)      extended knot vector
    k    : int            spline order

    Returns (B, d, G+k)
    """
    x_exp = x.unsqueeze(-1)           # (B, d, 1)
    g_exp = grid.view(1, 1, -1)       # (1, 1, n_knots)

    # Order-0: indicator over each knot interval
    b = ((x_exp >= g_exp[..., :-1]) & (x_exp < g_exp[..., 1:])).to(x.dtype)

    for p in range(1, k + 1):
        n_out = b.shape[-1] - 1

        t_i   = g_exp[..., :n_out]
        t_ip  = g_exp[..., p: p + n_out]
        left  = (x_exp - t_i) / (t_ip - t_i).clamp(min=1e-8)

        t_i1  = g_exp[..., 1: n_out + 1]
        t_ip1 = g_exp[..., p + 1: p + 1 + n_out]
        right = (t_ip1 - x_exp) / (t_ip1 - t_i1).clamp(min=1e-8)

        b = left * b[..., :n_out] + right * b[..., 1: n_out + 1]

    return b   # (B, d, n_basis)


# ── KAN layer ────────────────────────────────────────────────────────────────

class KANLayer(nn.Module):
    """
    One B-spline KAN layer (PINN-optimised).

        phi_{i->j}(x_i) = scale_sp_{ji}   * (spline_coef_{ji} . basis(x_i))
                        + scale_base_{ji} * silu(x_i)
        y_j = sum_i phi_{i->j} + bias_j

    Differences vs. the original implementation:
      * coef has a small random init (not zero)        -> stronger early gradients
      * scale_sp and scale_base are per-edge (out, in) -> matches pykan
      * forward avoids host-device sync                -> no .item() calls
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid: int = 5,
        k: int = 3,
        noise_scale: float = 0.1,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: Callable = F.silu,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid
        self.k            = k
        self.n_basis      = grid + k
        self.base_fun     = base_fun

        # Uniform internal grid on [-1, 1] + clamped extension on each side
        g = torch.linspace(-1.0, 1.0, grid + 1)
        self.register_buffer('grid', _extend_grid(g, k))

        # ── (1) Spline coefficients: SMALL RANDOM INIT (PINN-critical) ─────
        # Variance scaled by 1/sqrt(n_basis) so the summed spline output
        # stays O(noise_scale) at initialisation regardless of grid size.
        coef_init = (torch.rand(out_features, in_features, self.n_basis) - 0.5) \
                    * noise_scale / math.sqrt(self.n_basis)
        self.coef = nn.Parameter(coef_init)

        # ── (2) Per-edge residual weight (silu branch) ─────────────────────
        # pykan default: U(-1, 1) * sigma + mu, scaled by 1/sqrt(in_features).
        sb_init = scale_base_mu + scale_base_sigma * \
                  (torch.rand(out_features, in_features) * 2 - 1) / math.sqrt(in_features)
        self.scale_base = nn.Parameter(sb_init)

        # ── (3) Per-edge spline weight ─────────────────────────────────────
        ss_init = torch.ones(out_features, in_features) * scale_sp / math.sqrt(in_features)
        self.scale_sp = nn.Parameter(ss_init)

        # Per-output bias (stabilisation)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tensor-valued clamp -- no host-device sync, autograd safe
        x_c = torch.clamp(x, self.grid[self.k], self.grid[-(self.k + 1)])

        basis  = bspline_basis(x_c, self.grid, self.k)              # (B, in, n_basis)
        spline = torch.einsum('bin,oin->boi', basis, self.coef)     # (B, out, in)
        spline = spline * self.scale_sp.unsqueeze(0)                # (B, out, in)

        resid  = self.base_fun(x_c).unsqueeze(1) \
                 * self.scale_base.unsqueeze(0)                     # (B, out, in)

        return (spline + resid).sum(dim=2) + self.bias              # (B, out)

    # ── Optional: adaptive grid update (pykan trick) ────────────────────────
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01,
                    grid_eps: float = 0.02) -> None:
        """
        Reposition the (shared) knot vector based on the current input
        distribution, then re-fit the spline coefficients via least squares
        so the layer's output is preserved at the sample points.

        Call this OCCASIONALLY during training (e.g. every 500-2000 epochs)
        with a batch of representative inputs.  Not required, but it can
        squeeze a few extra percent of accuracy out of the model.

        x : (B, in_features) -- representative samples of the actual inputs
                                seen by this layer
        """
        device, dtype = x.device, x.dtype
        B  = x.shape[0]
        G  = self.grid_size
        k  = self.k

        # Aggregate quantiles across input dims (since we keep a 1-D shared grid)
        x_flat = x.reshape(-1)
        x_sorted, _ = torch.sort(x_flat)
        idx = torch.linspace(0, x_sorted.numel() - 1, G + 1,
                              device=device).long()
        quantile_grid = x_sorted[idx]

        # Blend with a uniform grid covering the data range -- robustness
        x_min = x_sorted[0] - margin
        x_max = x_sorted[-1] + margin
        uniform_grid = torch.linspace(x_min, x_max, G + 1,
                                       device=device, dtype=dtype)

        new_inner = (1.0 - grid_eps) * quantile_grid + grid_eps * uniform_grid
        new_inner, _ = torch.sort(new_inner)   # ensure monotonic
        new_grid = _extend_grid(new_inner, k)

        # ── Refit coefficients so f_old(x) ≈ f_new(x) ──────────────────────
        x_clamped_old = torch.clamp(x, self.grid[k], self.grid[-(k + 1)])
        basis_old = bspline_basis(x_clamped_old, self.grid, k)         # (B, in, n_basis)
        current   = torch.einsum('bin,oin->boi', basis_old, self.coef) # (B, out, in)

        x_clamped_new = torch.clamp(x, new_grid[k], new_grid[-(k + 1)])
        basis_new = bspline_basis(x_clamped_new, new_grid, k)          # (B, in, n_basis)

        # Solve (per input dim): basis_new[:, i, :] @ new_coef[o, i, :] = current[:, o, i]
        new_coef = torch.empty_like(self.coef)
        for i in range(self.in_features):
            A = basis_new[:, i, :]                                     # (B, n_basis)
            Y = current[:, :, i]                                       # (B, out)
            sol = torch.linalg.lstsq(A, Y).solution                    # (n_basis, out)
            new_coef[:, i, :] = sol.T

        # Commit
        self.grid.data = new_grid
        self.coef.data = new_coef


# ── Full KAN ─────────────────────────────────────────────────────────────────

class KAN(nn.Module):
    """
    Drop-in replacement for the original fast KAN.

        model = KAN(width=[2, 10, 10, 3], grid=25, k=5, device=device)
        model = model.to(torch.float64)
        y = model(x)   # x : (B, width[0])
    """

    def __init__(self, width: List[int], grid: int = 5, k: int = 3,
                 device: torch.device = torch.device('cpu'),
                 noise_scale: float = 0.1,
                 scale_base_mu: float = 0.0,
                 scale_base_sigma: float = 1.0,
                 base_fun=torch.nn.Tanh(),
                 scale_sp: float = 1.0):
        super().__init__()
        assert len(width) >= 2
        self.width = width
        self.grid  = grid
        self.k     = k

        self.layers = nn.ModuleList([
            KANLayer(width[i], width[i + 1], grid=grid, k=k,
                     noise_scale=noise_scale,
                     scale_base_mu=scale_base_mu,
                     scale_base_sigma=scale_base_sigma,
                     base_fun=base_fun,
                     scale_sp=scale_sp)
            for i in range(len(width) - 1)
        ])
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def update_grids(self, x: torch.Tensor) -> None:
        """Forward `x` once and update grids of every KAN layer.  Optional."""
        for layer in self.layers:
            layer.update_grid(x)
            x = layer(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = self.num_parameters()
        rows = [
            f"  KANLayer({l.in_features}->{l.out_features}, "
            f"grid={l.grid_size}, k={l.k}, "
            f"params={sum(p.numel() for p in l.parameters()):,})"
            for l in self.layers
        ]
        return (
            f"KAN(width={self.width}, grid={self.grid}, k={self.k})\n"
            + "\n".join(rows)
            + f"\n  Total trainable parameters: {total:,}"
        )
