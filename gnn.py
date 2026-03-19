from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from gudonov import solve_fvm


def flux(u: torch.Tensor) -> torch.Tensor:
    return u * (1.0 - u)


def godunov_flux_lwr(uL: torch.Tensor, uR: torch.Tensor) -> torch.Tensor:
    fL = flux(uL)
    fR = flux(uR)
    uc = 0.5

    shock_case = uL < uR
    shock_flux = torch.where(
        (uL <= uc) & (uc <= uR),
        flux(torch.full_like(uL, uc)),
        torch.minimum(fL, fR),
    )
    rare_flux = torch.maximum(fL, fR)
    return torch.where(shock_case, shock_flux, rare_flux)


def make_riemann_ic(x: np.ndarray, uL: float, uR: float, x_jump: float) -> np.ndarray:
    return np.where(x < x_jump, uL, uR).astype(np.float32)


class FluxMLP(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers = []
        for i in range(depth - 1):
            layers.append(nn.Linear(2 if i == 0 else hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, uL: torch.Tensor, uR: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([uL, uR], dim=-1))


class LearnedConservativeSolver(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3):
        super().__init__()
        self.flux_net = FluxMLP(hidden=hidden, depth=depth)

    def interface_fluxes(self, u: torch.Tensor) -> torch.Tensor:
        u_pad = torch.cat([u[:, :1], u, u[:, -1:]], dim=1)
        uL = u_pad[:, :-1].unsqueeze(-1)
        uR = u_pad[:, 1:].unsqueeze(-1)
        return self.flux_net(uL, uR).squeeze(-1)

    def step(self, u: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        fhat = self.interface_fluxes(u)
        return u - (dt / dx) * (fhat[:, 1:] - fhat[:, :-1])

    def rollout(self, u0: torch.Tensor, nsteps: int, dt: float, dx: float) -> torch.Tensor:
        traj = [u0]
        u = u0
        for _ in range(nsteps):
            u = self.step(u, dt, dx)
            traj.append(u)
        return torch.stack(traj, dim=0)


@dataclass
class Config:
    x_min: float = 0.0
    x_max: float = 1.0
    nx: int = 200
    t_max: float = 0.25
    cfl: float = 0.9
    x_jump: float = 0.5

    n_train: int = 64
    u_min: float = 0.05
    u_max: float = 0.95

    test_pair: tuple[float, float] = (0.9, 0.1)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden: int = 64
    depth: int = 3
    lr: float = 1e-3
    epochs: int = 5000
    print_every: int = 200
    seed: int = 0
    save_dir: str = "results_gnn_random_riemann"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_time_grid(cfg: Config):
    dx = (cfg.x_max - cfg.x_min) / cfg.nx
    dt = cfg.cfl * dx
    nsteps = int(math.ceil(cfg.t_max / dt))
    dt = cfg.t_max / nsteps
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, endpoint=False, dtype=np.float32)
    return x, dx, dt, nsteps


def call_godunov(u0: np.ndarray, cfg: Config):
    result = solve_fvm(
        u0=u0,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_max=cfg.t_max,
        cfl=cfg.cfl,
    )
    if isinstance(result, tuple) and len(result) == 3:
        _, _, u_hist = result
    elif isinstance(result, tuple) and len(result) == 2:
        _, u_hist = result
    else:
        raise ValueError("Unexpected solve_fvm output format.")
    return np.asarray(u_hist, dtype=np.float32)


def sample_train_pairs(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    pairs = []
    for _ in range(cfg.n_train):
        uL = rng.uniform(cfg.u_min, cfg.u_max)
        uR = rng.uniform(cfg.u_min, cfg.u_max)
        while abs(uL - uR) < 0.05:
            uR = rng.uniform(cfg.u_min, cfg.u_max)
        pairs.append((float(uL), float(uR)))
    return pairs


def build_dataset(cfg: Config, pairs):
    x, dx, dt, nsteps = build_time_grid(cfg)

    trajs = []
    for uL, uR in pairs:
        u0 = make_riemann_ic(x, uL, uR, cfg.x_jump)
        u_hist = call_godunov(u0, cfg)
        if len(u_hist) != nsteps + 1:
            m = min(len(u_hist) - 1, nsteps)
            u_hist = u_hist[:m + 1]
        trajs.append(u_hist)

    nsteps = min(len(tr) - 1 for tr in trajs)
    trajs = [tr[:nsteps + 1] for tr in trajs]

    u0_arr = np.stack([tr[0] for tr in trajs], axis=0)
    traj_arr = np.stack(trajs, axis=1)  # (T+1, N, nx)
    return x, dx, dt, nsteps, u0_arr, traj_arr


def plot_training(history, save_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmaps(x, dt, u_true, u_pred, save_path):
    err = np.abs(u_pred - u_true)
    t_max = dt * (len(u_true) - 1)
    vmin = min(np.min(u_true), np.min(u_pred))
    vmax = max(np.max(u_true), np.max(u_pred))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    im0 = axes[0].imshow(
        u_true, aspect="auto", origin="lower",
        extent=[x[0], x[-1], 0.0, t_max], vmin=vmin, vmax=vmax
    )
    axes[0].set_title("Truth (Godunov)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        u_pred, aspect="auto", origin="lower",
        extent=[x[0], x[-1], 0.0, t_max], vmin=vmin, vmax=vmax
    )
    axes[1].set_title("NN prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err, aspect="auto", origin="lower",
        extent=[x[0], x[-1], 0.0, t_max]
    )
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    cfg = Config()
    ensure_dir(cfg.save_dir)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    train_pairs = sample_train_pairs(cfg)
    print(f"Using {len(train_pairs)} random training Riemann problems.")
    print("First 10 train pairs:", train_pairs[:10])
    print("Test pair:", cfg.test_pair)

    x, dx, dt, nsteps, u0_train_np, traj_train_np = build_dataset(cfg, train_pairs)
    _, _, _, nsteps_test, u0_test_np, traj_test_np = build_dataset(cfg, [cfg.test_pair])

    nsteps = min(nsteps, nsteps_test)
    traj_train_np = traj_train_np[:nsteps + 1]
    traj_test_np = traj_test_np[:nsteps + 1]

    u0_train = torch.tensor(u0_train_np, dtype=torch.float32, device=device)
    traj_train = torch.tensor(traj_train_np, dtype=torch.float32, device=device)

    u0_test = torch.tensor(u0_test_np[:1], dtype=torch.float32, device=device)
    traj_test = torch.tensor(traj_test_np[:,:1], dtype=torch.float32, device=device)

    model = LearnedConservativeSolver(hidden=cfg.hidden, depth=cfg.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()

        pred_train = model.rollout(u0_train, nsteps, dt, dx)
        loss_rollout = torch.mean((pred_train - traj_train) ** 2)

        with torch.no_grad():
            u_pad = torch.cat([u0_train[:, :1], u0_train, u0_train[:, -1:]], dim=1)
            uL = u_pad[:, :-1].unsqueeze(-1)
            uR = u_pad[:, 1:].unsqueeze(-1)
            f_true = godunov_flux_lwr(uL, uR).squeeze(-1)

        f_pred = model.interface_fluxes(u0_train)
        loss_flux = torch.mean((f_pred - f_true) ** 2)

        loss = loss_rollout + 0.1 * loss_flux
        loss.backward()
        opt.step()

        history.append(float(loss.item()))

        if epoch % cfg.print_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred_test_tmp = model.rollout(u0_test, nsteps, dt, dx)
                test_mse = torch.mean((pred_test_tmp - traj_test) ** 2).item()
            print(
                f"Epoch {epoch:5d} | train={loss.item():.6e} | "
                f"rollout={loss_rollout.item():.6e} | flux={loss_flux.item():.6e} | "
                f"test={test_mse:.6e}"
            )

    print(f"\nTraining done in {time.time() - t0:.2f} s")

    model.eval()
    with torch.no_grad():
        pred_test = model.rollout(u0_test, nsteps, dt, dx).cpu().numpy()

    test_true = traj_test.cpu().numpy()[:, 0, :]
    test_pred = pred_test[:, 0, :]

    plot_training(history, cfg.save_dir)
    plot_heatmaps(
        x, dt, test_true, test_pred,
        os.path.join(cfg.save_dir, "test_random_train_heatmaps.png")
    )

    np.savez(
        os.path.join(cfg.save_dir, "results_random_train.npz"),
        x=x,
        dt=dt,
        train_pairs=np.array(train_pairs, dtype=np.float32),
        test_pair=np.array(cfg.test_pair, dtype=np.float32),
        u_true=test_true,
        u_pred=test_pred,
        history=np.array(history, dtype=np.float32),
    )

    print(f"Saved to {cfg.save_dir}")


if __name__ == "__main__":
    main()