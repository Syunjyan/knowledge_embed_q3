"""Physics residuals for official Transolver cases. Attention is not touched."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def lock_scale(epoch: int, epochs: int, group: str, unlock: float, lock: float) -> float:
    """A/HARD: no extra loss. C/MATCH/POS/CONS: constant unlock. D: late lock. DYN: unused."""
    group = group.upper()
    if group in {"A", "HARD", "S100A"}:
        return 0.0
    if group in {"C", "MATCH", "POS", "CONS", "S100M", "S100C"}:
        return unlock
    if group in {"D", "DYN"}:
        return lock if epoch >= epochs // 2 else unlock
    raise ValueError(f"unknown group {group}")


def dynamic_weight(l_data: Tensor, l_phys: Tensor, cap: float = 0.1, frac: float = 0.05) -> Tensor:
    """Keep physics from dominating data. Plan item: dynamic loss weighting / conflict control."""
    phys = l_phys.detach()
    data = l_data.detach()
    w = frac * data / (phys + 1e-8)
    return torch.clamp(w, min=0.0, max=cap)


def central_diff(x: Tensor, h: float, resolution: int) -> tuple[Tensor, Tensor]:
    x = rearrange(x, "b (h w) c -> b h w c", h=resolution, w=resolution)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode="constant", value=0.0)
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)
    return grad_x, grad_y


def dirichlet_mask(s: int, device, dtype) -> Tensor:
    m = torch.ones(s, s, device=device, dtype=dtype)
    m[0, :] = 0
    m[-1, :] = 0
    m[:, 0] = 0
    m[:, -1] = 0
    return m


def apply_dirichlet(u: Tensor, s: int) -> Tensor:
    """Hard Dirichlet u=0 on ∂Ω, i.e. BC as structure rather than a competing residual."""
    bsz = u.shape[0]
    mask = dirichlet_mask(s, u.device, u.dtype).view(1, s, s)
    return (u.reshape(bsz, s, s) * mask).reshape(bsz, -1)


def darcy_residual_field(u: Tensor, a: Tensor, dx: float, s: int) -> Tensor:
    """Interior field of -div(a ∇u) - 1. Fourier flux + energy balance on Darcy/heat-like elliptic PDE."""
    bsz = u.shape[0]
    u_img = rearrange(u.reshape(bsz, -1, 1), "b (h w) c -> b c h w", h=s)
    u_img = F.pad(u_img[..., 1:-1, 1:-1].contiguous(), (1, 1, 1, 1), "constant", 0)
    u_flat = rearrange(u_img, "b c h w -> b (h w) c")
    gx, gy = central_diff(u_flat, dx, s)
    a_grid = a.reshape(bsz, s, s, 1)
    flux_x = a_grid * gx
    flux_y = a_grid * gy
    dfx_dx, _ = central_diff(rearrange(flux_x, "b h w c -> b (h w) c"), dx, s)
    _, dfy_dy = central_diff(rearrange(flux_y, "b h w c -> b (h w) c"), dx, s)
    return (-(dfx_dx + dfy_dy).squeeze(-1) - 1.0)[:, 1:-1, 1:-1]


def darcy_pde_residual(u: Tensor, a: Tensor, dx: float, s: int) -> Tensor:
    """Strong form ||R(u,a)||^2. Failed at full labels: discrete R(u*) ≠ 0."""
    return torch.mean(darcy_residual_field(u, a, dx, s) ** 2)


def darcy_pde_match(u_pred: Tensor, u_gt: Tensor, a: Tensor, dx: float, s: int) -> Tensor:
    """Match discrete residual of prediction to that of labels. PDE in loss without fighting FD error."""
    r_pred = darcy_residual_field(u_pred, a, dx, s)
    r_gt = darcy_residual_field(u_gt, a, dx, s)
    return torch.mean((r_pred - r_gt) ** 2)


def positivity_residual(u: Tensor) -> Tensor:
    """Constitutive inequality: von Mises / pressure-like scalar is non-negative."""
    return torch.mean(F.relu(-u) ** 2)


def ns_vorticity_residual(w: Tensor, w_prev: Tensor, h: int, dt: float = 1.0, nu: float = 1e-5) -> Tensor:
    """2D incompressible vorticity form. Kept for ablation; not used in the rescue queue."""
    w = w.reshape(-1, h, h)
    w_prev = w_prev.reshape(-1, h, h)
    device = w.device
    dtype = w.dtype
    kx = 2.0 * torch.pi * torch.fft.fftfreq(h, d=1.0 / h, device=device, dtype=dtype)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(h, d=1.0 / h, device=device, dtype=dtype)
    ky, kx = torch.meshgrid(ky, kx, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0
    w_hat = torch.fft.fft2(w)
    psi_hat = w_hat / k2_safe
    psi_hat[..., 0, 0] = 0
    u = torch.fft.ifft2(1j * ky * psi_hat).real
    v = torch.fft.ifft2(-1j * kx * psi_hat).real
    wx = torch.fft.ifft2(1j * kx * w_hat).real
    wy = torch.fft.ifft2(1j * ky * w_hat).real
    lap = torch.fft.ifft2(-k2 * w_hat).real
    res = (w - w_prev) / dt + u * wx + v * wy - nu * lap
    return torch.mean(res ** 2) / (torch.mean(w ** 2) + 1e-8)


def ns_mean_conservation(w: Tensor, w_prev: Tensor) -> Tensor:
    """Periodic incompressible NS: spatial mean vorticity is conserved (Galilean / integral symmetry)."""
    w = w.reshape(w.shape[0], -1)
    w_prev = w_prev.reshape(w_prev.shape[0], -1)
    return torch.mean((w.mean(dim=-1) - w_prev.mean(dim=-1)) ** 2) / (torch.mean(w ** 2) + 1e-8)
