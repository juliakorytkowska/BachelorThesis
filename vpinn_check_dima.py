from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import torch

import config as cfg
from vpinn_dima import LWRVPINN


# ============================================================
# 1) Single-jump initial condition
# ============================================================
UL = 0.1
UR = 0.3
X0 = 0.0


def u0_single_jump_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < X0, torch.full_like(x, UL), torch.full_like(x, UR))


def u0_single_jump_np(x: np.ndarray) -> np.ndarray:
    return np.where(x < X0, UL, UR)


# ============================================================
# 2) Godunov solver for LWR: f(u)=u(1-u)
# ============================================================
def flux_np(u: np.ndarray) -> np.ndarray:
    return u * (1.0 - u)


def godunov_flux_lwr(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
    """
    Godunov flux for concave LWR flux f(u)=u(1-u), u in [0,1].
    """
    fL = flux_np(uL)
    fR = flux_np(uR)
    fmax = 0.25

    out = np.empty_like(uL)

    # For concave flux:
    # shock if uL < uR
    shock = uL < uR
    s = 1.0 - (uL + uR)

    left_shock = shock & (s >= 0.0)
    right_shock = shock & (s < 0.0)
    out[left_shock] = fL[left_shock]
    out[right_shock] = fR[right_shock]

    # rarefaction if uL >= uR
    rare = ~shock
    all_left = rare & (uL <= 0.5)
    all_right = rare & (uR >= 0.5)
    through_max = rare & ~(all_left | all_right)

    out[all_left] = fL[all_left]
    out[all_right] = fR[all_right]
    out[through_max] = fmax

    return out


def godunov_lwr(
    nx: int = 400,
    t_final: float = 1.0,
    cfl: float = 0.9,
):
    dx = (cfg.X_MAX - cfg.X_MIN) / nx
    x = cfg.X_MIN + (np.arange(nx) + 0.5) * dx

    u = u0_single_jump_np(x).astype(np.float64)
    t = 0.0

    U_hist = [u.copy()]
    t_hist = [t]

    while t < t_final - 1e-14:
        speed = np.max(np.abs(1.0 - 2.0 * u))
        speed = max(speed, 1e-8)
        dt = cfl * dx / speed
        if t + dt > t_final:
            dt = t_final - t

        u_ext = np.empty(nx + 2, dtype=np.float64)
        u_ext[1:-1] = u
        u_ext[0] = UL
        u_ext[-1] = UR

        F = godunov_flux_lwr(u_ext[:-1], u_ext[1:])
        u = u - (dt / dx) * (F[1:] - F[:-1])

        t += dt
        U_hist.append(u.copy())
        t_hist.append(t)

    return x, np.array(t_hist), np.array(U_hist)


# ============================================================
# 3) Sampling helpers
# ============================================================
def sample_x(n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN


def sample_t(n: int, device: torch.device, bias_early: bool = True) -> torch.Tensor:
    if bias_early:
        z = torch.rand(n, 1, device=device)
        return cfg.T_MIN + (cfg.T_MAX - cfg.T_MIN) * (z ** 2)
    return torch.rand(n, 1, device=device) * (cfg.T_MAX - cfg.T_MIN) + cfg.T_MIN


def sample_initial(n: int, device: torch.device):
    x = sample_x(n, device)
    t = torch.full_like(x, cfg.T_MIN)
    return x, t


# ============================================================
# 4) Losses
# ============================================================
def ic_loss(model: LWRVPINN, n: int, device: torch.device) -> torch.Tensor:
    x0, t0 = sample_initial(n, device)
    u_true = u0_single_jump_torch(x0)
    u_pred = model(x0, t0, u0_fn=u0_single_jump_torch)
    return torch.mean((u_pred - u_true) ** 2)


def bc_loss(model: LWRVPINN, n: int, device: torch.device) -> torch.Tensor:
    t = sample_t(n, device, bias_early=False)
    xL = torch.full_like(t, cfg.X_MIN)
    xR = torch.full_like(t, cfg.X_MAX)

    uL = model(xL, t, u0_fn=u0_single_jump_torch)
    uR = model(xR, t, u0_fn=u0_single_jump_torch)

    lossL = torch.mean((uL - torch.full_like(uL, UL)) ** 2)
    lossR = torch.mean((uR - torch.full_like(uR, UR)) ** 2)
    return lossL + lossR


@dataclass
class TrainHistory:
    total: list
    weak: list
    ic: list
    bc: list


# ============================================================
# 5) Training
# ============================================================
def train_vpinn_single_jump(
    device: torch.device,
    steps: int = 20000,
    lr: float = 1e-3,
    n_weak: int = 4096,
    n_ic: int = 2048,
    n_bc: int = 1024,
    w_weak: float = 1.0,
    w_ic: float = 50.0,
    w_bc: float = 10.0,
):
    model = LWRVPINN(
        layers=[128, 128, 128, 128],
        activation="tanh",
        hard_init=True,
        n_fourier=6,
        scale=2.0,
        n_test_x=4,
        n_test_t=4,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = TrainHistory(total=[], weak=[], ic=[], bc=[])

    for it in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        x = sample_x(n_weak, device)
        t = sample_t(n_weak, device, bias_early=True)

        loss_weak = model.weak_residual_loss(x, t, u0_fn=u0_single_jump_torch)
        loss_ic = ic_loss(model, n_ic, device)
        loss_bc = bc_loss(model, n_bc, device)

        loss = w_weak * loss_weak + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        opt.step()

        if it == 1 or it % 500 == 0:
            lw = float(loss_weak.detach().cpu())
            li = float(loss_ic.detach().cpu())
            lb = float(loss_bc.detach().cpu())
            lt = float(loss.detach().cpu())

            print(
                f"[{it:6d}/{steps}] total={lt:.3e} | weak={lw:.3e} | ic={li:.3e} | bc={lb:.3e}"
            )

            hist.total.append(lt)
            hist.weak.append(lw)
            hist.ic.append(li)
            hist.bc.append(lb)

    return model, hist


# ============================================================
# 6) Evaluation
# ============================================================
@torch.no_grad()
def evaluate_model(model: LWRVPINN, x_grid: np.ndarray, t_grid: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X, T = np.meshgrid(x_grid, t_grid)

    xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)

    up = model(xt, tt, u0_fn=u0_single_jump_torch)
    return up.cpu().numpy().reshape(len(t_grid), len(x_grid))


def interp_time_snapshots(U_hist: np.ndarray, t_hist: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_eval), U_hist.shape[1]), dtype=np.float64)

    for i, t in enumerate(t_eval):
        j = np.searchsorted(t_hist, t) - 1
        j = np.clip(j, 0, len(t_hist) - 2)

        t0 = t_hist[j]
        t1 = t_hist[j + 1]
        w = (t - t0) / (t1 - t0 + 1e-12)

        out[i] = (1.0 - w) * U_hist[j] + w * U_hist[j + 1]

    return out


# ============================================================
# 7) Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # Godunov reference
    xg, tg, Ug = godunov_lwr(nx=400, t_final=cfg.T_MAX, cfl=0.9)

    # Train VPINN
    model, hist = train_vpinn_single_jump(
        device=device,
        steps=3000,
        lr=1e-3,
        n_weak=4096,
        n_ic=2048,
        n_bc=1024,
        w_weak=1.0,
        w_ic=50.0,
        w_bc=10.0,
    )

    # Evaluate on same x grid
    t_eval = np.linspace(cfg.T_MIN, cfg.T_MAX, 101)
    U_vpinn = evaluate_model(model, xg, t_eval, device)
    U_god = interp_time_snapshots(Ug, tg, t_eval)

    mse = np.mean((U_vpinn - U_god) ** 2)
    print(f"\nMSE(VPINN, Godunov) = {mse:.6e}")

    # --------------------------------------------------------
    # Snapshot plots
    # --------------------------------------------------------
    times_to_plot = [0.0, 0.2, 0.5, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, tstar in zip(axes, times_to_plot):
        idx = np.argmin(np.abs(t_eval - tstar))
        ax.plot(xg, U_god[idx], label="Godunov", linewidth=2)
        ax.plot(xg, U_vpinn[idx], "--", label="VPINN", linewidth=2)
        ax.set_title(f"t = {t_eval[idx]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True)
        ax.legend()

    plt.show()

    # --------------------------------------------------------
    # Heatmaps
    # --------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    im0 = axes2[0].imshow(
        U_god,
        extent=[xg[0], xg[-1], t_eval[-1], t_eval[0]],
        aspect="auto",
    )
    axes2[0].set_title("Godunov")
    axes2[0].set_xlabel("x")
    axes2[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes2[0])

    im1 = axes2[1].imshow(
        U_vpinn,
        extent=[xg[0], xg[-1], t_eval[-1], t_eval[0]],
        aspect="auto",
    )
    axes2[1].set_title("VPINN")
    axes2[1].set_xlabel("x")
    axes2[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes2[1])

    im2 = axes2[2].imshow(
        np.abs(U_vpinn - U_god),
        extent=[xg[0], xg[-1], t_eval[-1], t_eval[0]],
        aspect="auto",
    )
    axes2[2].set_title("|VPINN - Godunov|")
    axes2[2].set_xlabel("x")
    axes2[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes2[2])
    plt.savefig(f"dima_snapshots.png", dpi=300)


    plt.show()


if __name__ == "__main__":
    main()