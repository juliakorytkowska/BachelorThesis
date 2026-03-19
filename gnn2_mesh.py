from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from gudonov import godunov_flux, add_ghost_cells


# ============================================================
# 1) LWR physics
# ============================================================
def flux_lwr_torch(u: torch.Tensor) -> torch.Tensor:
    return u * (1.0 - u)


def godunov_flux_torch(u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
    f_left = flux_lwr_torch(u_left)
    f_right = flux_lwr_torch(u_right)

    f_min = torch.minimum(f_left, f_right)
    f_max = torch.maximum(f_left, f_right)

    u_lo = torch.minimum(u_left, u_right)
    u_hi = torch.maximum(u_left, u_right)
    has_mid = (u_lo <= 0.5) & (0.5 <= u_hi)
    f_max = torch.where(has_mid, torch.maximum(f_max, f_max.new_full((), 0.25)), f_max)

    return torch.where(u_left <= u_right, f_min, f_max)


def make_riemann_ic(
    x: np.ndarray, u_left: float, u_right: float, x_jump: float = 0.0
) -> np.ndarray:
    return np.where(x < x_jump, u_left, u_right).astype(np.float32)


# ============================================================
# 2) Truth solver with fixed dt
# ============================================================
def solve_truth_fixed_dt(
    u0: np.ndarray,
    dx: float,
    dt: float,
    t_max: float,
    bc: str = "copy",
) -> Tuple[np.ndarray, np.ndarray]:
    u = u0.copy().astype(float)
    t = 0.0
    times = [0.0]
    states = [u.copy()]

    while t < t_max - 1e-14:
        dt_step = min(dt, t_max - t)
        u_ext = add_ghost_cells(u, bc=bc)
        fhat = godunov_flux(u_ext)
        u = u - (dt_step / dx) * (fhat[1:] - fhat[:-1])
        t += dt_step
        times.append(t)
        states.append(u.copy())

    return np.array(times), np.stack(states, axis=0)


# ============================================================
# 3) Small MLP helper
# ============================================================
def make_mlp(
    in_dim: int, hidden: int, out_dim: int, depth: int, activation: str
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    act_cls = nn.GELU if activation.lower() == "gelu" else nn.Tanh
    dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]

    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


# ============================================================
# 4) FluxGNN 1D latent model
# ============================================================
class FluxGNN1DLatent(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden: int = 64,
        depth: int = 3,
        activation: str = "gelu",
        use_base_flux: bool = False,
        base_flux_weight: float = 0.5,
        latent_flux_scale: float = 0.25,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.use_base_flux = bool(use_base_flux)
        self.base_flux_weight = float(base_flux_weight)
        self.latent_flux_scale = float(latent_flux_scale)

        e = torch.ones(self.latent_dim, dtype=torch.float32) / np.sqrt(self.latent_dim)
        self.encoder_vec = nn.Parameter(e)

        in_dim = 2 * self.latent_dim
        self.flux_mlp = make_mlp(in_dim, hidden, self.latent_dim, depth, activation)

    def decoder_vec(self) -> torch.Tensor:
        e = self.encoder_vec
        return e / (torch.sum(e * e) + 1e-12)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return u.unsqueeze(-1) * self.encoder_vec

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sum(h * self.decoder_vec(), dim=-1)

    def build_edge_features(self, h_left: torch.Tensor, h_right: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_left + h_right, torch.abs(h_right - h_left)], dim=-1)

    def latent_neural_flux(self, h_left: torch.Tensor, h_right: torch.Tensor) -> torch.Tensor:
        edge_feat = self.build_edge_features(h_left, h_right)
        flat = edge_feat.reshape(-1, edge_feat.shape[-1])
        out = self.flux_mlp(flat).reshape_as(h_left)
        return torch.tanh(out) * self.latent_flux_scale

    def compute_latent_flux(self, u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
        h_left = self.encode(u_left)
        h_right = self.encode(u_right)
        flux_learned = self.latent_neural_flux(h_left, h_right)

        if self.use_base_flux:
            flux_base_latent = self.encode(godunov_flux_torch(u_left, u_right))
            w = self.base_flux_weight
            return (1.0 - w) * flux_base_latent + w * flux_learned

        return flux_learned

    def step(self, u: torch.Tensor, dt: float, dx: float, boundary: str = "copy") -> torch.Tensor:
        h = self.encode(u)

        if boundary == "copy":
            u_ext = torch.empty(u.size(0), u.size(1) + 2, device=u.device, dtype=u.dtype)
            u_ext[:, 1:-1] = u
            u_ext[:, 0] = u[:, 0]
            u_ext[:, -1] = u[:, -1]
            flux = self.compute_latent_flux(u_ext[:, :-1], u_ext[:, 1:])
            h_new = h - (dt / dx) * (flux[:, 1:] - flux[:, :-1])
            return self.decode(h_new)

        if boundary == "periodic":
            u_right = torch.roll(u, shifts=-1, dims=1)
            flux = self.compute_latent_flux(u, u_right)
            h_new = h - (dt / dx) * (flux - torch.roll(flux, shifts=1, dims=1))
            return self.decode(h_new)

        if boundary == "fixed":
            flux = self.compute_latent_flux(u[:, :-1], u[:, 1:])
            h_new = h.clone()
            h_new[:, 1:-1] = h[:, 1:-1] - (dt / dx) * (flux[:, 1:] - flux[:, :-1])
            u_new = self.decode(h_new)
            u_new[:, 0] = u[:, 0]
            u_new[:, -1] = u[:, -1]
            return u_new

        raise ValueError("boundary must be 'periodic', 'copy', or 'fixed'")

    def rollout(
        self, u0: torch.Tensor, dt: float, dx: float, n_steps: int, boundary: str = "copy"
    ) -> torch.Tensor:
        u = u0
        outs = [u]
        for _ in range(1, n_steps):
            u = self.step(u, dt, dx, boundary)
            outs.append(u)
        return torch.stack(outs, dim=1)


# ============================================================
# 5) Config
# ============================================================
@dataclass
class Config:
    x_min: float = -1.0
    x_max: float = 1.0
    t_max: float = 0.8
    cfl: float = 0.45
    boundary: str = "copy"

    # train on coarse mesh, test on finer mesh
    nx_train: int = 128
    nx_test: int = 256

    train_pairs: Tuple[Tuple[float, float], ...] = (
        (0.75, 0.20),
        (0.35, 0.80),
        (0.60, 0.10),
    )
    x_jump_train: float = 0.0

    test_pair: Tuple[float, float] = (0.65, 0.15)
    x_jump_test: float = 0.0

    latent_dim: int = 32
    hidden: int = 64
    depth: int = 3
    activation: str = "gelu"
    use_base_flux: bool = False
    base_flux_weight: float = 0.50
    latent_flux_scale: float = 0.25

    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    seed: int = 0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "results_fluxgnn_mesh_transfer"


# ============================================================
# 6) Grid and dt
# ============================================================
def build_grid(x_min: float, x_max: float, nx: int) -> Tuple[np.ndarray, float]:
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    dx = float(x[1] - x[0])
    return x, dx


def build_fixed_dt(cfl: float, dx: float) -> float:
    return cfl * dx / 1.0


# ============================================================
# 7) Dataset generation
# ============================================================
def build_train_dataset(
    cfg: Config, x: np.ndarray, dx: float, dt: float
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    u0_list, y_list = [], []
    times_ref = None

    for uL, uR in cfg.train_pairs:
        u0 = make_riemann_ic(x, uL, uR, cfg.x_jump_train)
        times, U = solve_truth_fixed_dt(
            u0, dx=dx, dt=dt, t_max=cfg.t_max, bc=cfg.boundary
        )

        if times_ref is None:
            times_ref = times
        else:
            assert len(times) == len(times_ref)

        u0_list.append(u0)
        y_list.append(U.astype(np.float32))

    u0_train = torch.from_numpy(np.stack(u0_list, axis=0))
    y_train = torch.from_numpy(np.stack(y_list, axis=0))
    return u0_train, y_train, times_ref


# ============================================================
# 8) Training
# ============================================================
def train_model(
    cfg: Config,
    model: FluxGNN1DLatent,
    u0_train: torch.Tensor,
    y_train: torch.Tensor,
    dt: float,
    dx: float,
) -> List[float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    u0_train = u0_train.to(cfg.device)
    y_train = y_train.to(cfg.device)
    n_steps = y_train.shape[1]

    loss_hist: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)

        pred = model.rollout(u0_train, dt=dt, dx=dx, n_steps=n_steps, boundary=cfg.boundary)
        loss = torch.mean((pred - y_train) ** 2)

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        loss_hist.append(float(loss.item()))

        if epoch == 1 or epoch % 100 == 0:
            print(f"epoch {epoch:5d} | train MSE = {loss.item():.8e}")

    return loss_hist


# ============================================================
# 9) Evaluation on a different mesh
# ============================================================
def evaluate_mesh_transfer(
    cfg: Config,
    model: FluxGNN1DLatent,
    x: np.ndarray,
    dx: float,
    dt: float,
):
    uL, uR = cfg.test_pair
    u0_test = make_riemann_ic(x, uL, uR, cfg.x_jump_test)

    times_test, U_true = solve_truth_fixed_dt(
        u0_test, dx=dx, dt=dt, t_max=cfg.t_max, bc=cfg.boundary
    )
    U_true = U_true.astype(np.float32)

    model.eval()
    with torch.no_grad():
        u0_t = torch.from_numpy(u0_test[None, :]).to(cfg.device)
        U_pred = model.rollout(
            u0_t, dt=dt, dx=dx, n_steps=len(times_test), boundary=cfg.boundary
        )[0].detach().cpu().numpy()

    nt = min(U_true.shape[0], U_pred.shape[0])
    U_true = U_true[:nt]
    U_pred = U_pred[:nt]
    times_test = times_test[:nt]

    err = U_pred - U_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    linf = float(np.max(np.abs(err)))

    print("\nMesh-transfer test")
    print(f"  trained on nx = {cfg.nx_train}")
    print(f"  tested  on nx = {cfg.nx_test}")
    print(f"  pair = ({uL:.3f}, {uR:.3f})")
    print(f"  MSE  = {mse:.8e}")
    print(f"  MAE  = {mae:.8e}")
    print(f"  Linf = {linf:.8e}")

    return u0_test, U_true, U_pred, err, times_test, mse, mae, linf


# ============================================================
# 10) Plotting
# ============================================================
def plot_final_snapshot(save_dir: str, x, U_true, U_pred):
    plt.figure(figsize=(8, 5))
    plt.plot(x, U_true[-1], label="Godunov truth", linewidth=2)
    plt.plot(x, U_pred[-1], "--", label="FluxGNN pred", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x,t_max)")
    plt.title("Mesh-transfer: final snapshot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mesh_transfer_final_snapshot.png"), dpi=160)
    plt.close()


def plot_space_time(save_dir: str, x, times, U_true, U_pred, err):
    nt = min(len(times), U_true.shape[0], U_pred.shape[0])
    extent = [x[0], x[-1], times[0], times[nt - 1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(U_true[:nt], aspect="auto", extent=extent, origin="lower", cmap="viridis")
    axes[0].set_title("Godunov truth")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(U_pred[:nt], aspect="auto", extent=extent, origin="lower", cmap="viridis")
    axes[1].set_title("FluxGNN prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.abs(err[:nt]), aspect="auto", extent=extent, origin="lower", cmap="hot", vmin=0)
    axes[2].set_title("|Prediction error|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mesh_transfer_spacetime.png"), dpi=160)
    plt.close()


# ============================================================
# 11) Main
# ============================================================
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # train mesh
    x_train, dx_train = build_grid(cfg.x_min, cfg.x_max, cfg.nx_train)
    dt_train = build_fixed_dt(cfg.cfl, dx_train)

    # test mesh
    x_test, dx_test = build_grid(cfg.x_min, cfg.x_max, cfg.nx_test)
    dt_test = build_fixed_dt(cfg.cfl, dx_test)

    print("Config")
    print(f"  device   : {cfg.device}")
    print(f"  nx_train : {cfg.nx_train}")
    print(f"  nx_test  : {cfg.nx_test}")
    print(f"  t_max    : {cfg.t_max}")
    print(f"  test pair: {cfg.test_pair}")

    u0_train, y_train, _ = build_train_dataset(cfg, x_train, dx_train, dt_train)

    model = FluxGNN1DLatent(
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        depth=cfg.depth,
        activation=cfg.activation,
        use_base_flux=cfg.use_base_flux,
        base_flux_weight=cfg.base_flux_weight,
        latent_flux_scale=cfg.latent_flux_scale,
    ).to(cfg.device)

    train_model(cfg, model, u0_train, y_train, dt_train, dx_train)

    u0_test, U_true, U_pred, err, times_test, mse, mae, linf = evaluate_mesh_transfer(
        cfg, model, x_test, dx_test, dt_test
    )

    with open(os.path.join(cfg.save_dir, "mesh_transfer_metrics.txt"), "w") as f:
        f.write(f"trained on nx = {cfg.nx_train}\n")
        f.write(f"tested on nx  = {cfg.nx_test}\n")
        f.write(f"test pair     = {cfg.test_pair}\n")
        f.write(f"MSE  = {mse:.12e}\n")
        f.write(f"MAE  = {mae:.12e}\n")
        f.write(f"Linf = {linf:.12e}\n")

    plot_final_snapshot(cfg.save_dir, x_test, U_true, U_pred)
    plot_space_time(cfg.save_dir, x_test, times_test, U_true, U_pred, err)

    print("\nDone.")


if __name__ == "__main__":
    main()