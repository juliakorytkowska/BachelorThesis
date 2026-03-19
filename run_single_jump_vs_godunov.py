# run_onejump_vpinn_vs_godunov.py
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from gudonov import solve_fvm
from vpinns import train_vpinn


# -------------------------
# One-jump (Riemann) IC utilities
# -------------------------
@dataclass
class OneJumpConfig:
    x0: float = 0.0     # jump location
    uL: float = 0.1     # left state
    uR: float = 0.3     # right state


def piecewise_constant_u0(x: np.ndarray, x0: float, uL: float, uR: float) -> np.ndarray:
    return np.where(x < x0, uL, uR).astype(float)


def make_u0_fns(x0: float, uL: float, uR: float):
    """Return numpy and torch versions of u0(x) for a single jump."""
    def u0_numpy(x: np.ndarray) -> np.ndarray:
        return piecewise_constant_u0(x, x0=x0, uL=uL, uR=uR)

    def u0_torch(x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x < x0,
            torch.tensor(float(uL), device=x.device),
            torch.tensor(float(uR), device=x.device),
        )

    return u0_numpy, u0_torch


def eval_on_grid(model, u0_torch, x: np.ndarray, t: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X, T = np.meshgrid(x, t, indexing="xy")
        xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
        tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
        u = model(xt, tt, u0_fn=u0_torch).reshape(len(t), len(x)).cpu().numpy()
    return u


# -------------------------
# Main experiment (one case)
# -------------------------
def run_one_case(cfg1: OneJumpConfig, device: torch.device) -> None:
    outdir = "results/onejump"
    os.makedirs(outdir, exist_ok=True)

    # Evaluation / truth grid
    nx = cfg.NX_EVAL
    x = np.linspace(cfg.X_MIN, cfg.X_MAX, nx)
    u0_numpy, u0_torch = make_u0_fns(cfg1.x0, cfg1.uL, cfg1.uR)

    # Godunov truth
    x_t, t_t, u_truth = solve_fvm(
        u0=u0_numpy(x),
        nt_out=cfg.NT_EVAL,
        x_min=cfg.X_MIN,
        x_max=cfg.X_MAX,
        t_max=cfg.T_MAX,
        cfl=0.3,
        bc="copy",
    )

    # Train VPINN (instance-specific)
    layers = [128, 128, 128]
    model, hist = train_vpinn(
        layers=layers,
        u0_fn=u0_torch,
        uL=float(cfg1.uL),
        uR=float(cfg1.uR),
        steps=cfg.STEPS,
        lr=cfg.LR,
        interior_samples=cfg.INTERIOR_SAMPLES,
        boundary_samples=cfg.BOUNDARY_SAMPLES,
        initial_samples=cfg.INITIAL_SAMPLES,
        hard_init=False,
        activation="tanh",
        n_test=cfg.N_TEST,
        device=device,
        log_every=cfg.LOG_EVERY,
    )

    # Predict
    u_pred = eval_on_grid(model, u0_torch, x_t, t_t, device)
    err = np.abs(u_pred - u_truth)

    # Metrics at final time
    rel_l2 = np.linalg.norm(u_pred[-1] - u_truth[-1]) / (np.linalg.norm(u_truth[-1]) + 1e-12)
    linf = np.max(np.abs(u_pred[-1] - u_truth[-1]))
    print(f"[one-jump] relL2={rel_l2:.3e} | Linf={linf:.3e} | x0={cfg1.x0} | uL={cfg1.uL} | uR={cfg1.uR}")

    # Save triptych
    f = os.path.join(outdir, "onejump_pred_truth_error.png")
    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plt.title("Truth (Godunov)")
    plt.pcolormesh(x_t, t_t, u_truth, shading="auto")
    plt.xlabel("x"); plt.ylabel("t")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("VPINN prediction")
    plt.pcolormesh(x_t, t_t, u_pred, shading="auto")
    plt.xlabel("x"); plt.ylabel("t")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Absolute error")
    plt.pcolormesh(x_t, t_t, err, shading="auto")
    plt.xlabel("x"); plt.ylabel("t")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f, dpi=200)
    print("Saved:", f)

    # Save initial condition plot
    f_ic = os.path.join(outdir, "onejump_u0.png")
    plt.figure(figsize=(6, 3))
    plt.title("Initial condition (one jump)")
    plt.plot(x, u0_numpy(x))
    plt.xlabel("x"); plt.ylabel("u(x,0)")
    plt.tight_layout()
    plt.savefig(f_ic, dpi=200)
    print("Saved:", f_ic)

    # Save final-time slice
    f_slice = os.path.join(outdir, "onejump_final_slice.png")
    plt.figure(figsize=(7, 4))
    plt.title(f"Final time slice (t={t_t[-1]:.2f})")
    plt.plot(x_t, u_truth[-1], label="Truth")
    plt.plot(x_t, u_pred[-1], label="VPINN")
    plt.legend()
    plt.xlabel("x"); plt.ylabel("u(x,t)")
    plt.tight_layout()
    plt.savefig(f_slice, dpi=200)
    print("Saved:", f_slice)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cfg1 = OneJumpConfig(x0=0.0, uL=0.1, uR=0.3)
    run_one_case(cfg1, device=device)


if __name__ == "__main__":
    main()