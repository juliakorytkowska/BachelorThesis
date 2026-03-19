# train_fno_2.py
# ============================================================
# Train FNO on Godunov data for u_t + (u(1-u))_x = 0
# Single-jump piecewise constant ICs
# x horizontal, t vertical in plots
# ============================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from gudonov import solve_fvm


# ----------------------------
# 1) Spectral convolution
# ----------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(scale * torch.complex(
            torch.randn(in_channels, out_channels, modes1, modes2),
            torch.randn(in_channels, out_channels, modes1, modes2)
        ))
        self.weights2 = nn.Parameter(scale * torch.complex(
            torch.randn(in_channels, out_channels, modes1, modes2),
            torch.randn(in_channels, out_channels, modes1, modes2)
        ))

    def forward(self, x):
        # x: (B, C, nx, nt)
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            B, x.shape[1], x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


# ----------------------------
# 2) FNO model
# ----------------------------
class FNO_SpaceTime_Final(nn.Module):
    def __init__(self, modes1=32, modes2=32, width=96, layers=4):
        super().__init__()
        self.fc0 = nn.Linear(3, width)

        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(layers)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, nx, nt, 3)
        x = self.fc0(x)                 # (B, nx, nt, width)
        x = x.permute(0, 3, 1, 2)       # (B, width, nx, nt)

        for conv, w, bn in zip(self.convs, self.ws, self.bns):
            x = bn(conv(x) + w(x))
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)       # (B, nx, nt, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)     # (B, nx, nt)
        return x


# ----------------------------
# 3) Data generation (fresh each run)
# ----------------------------
def generate_pde_data(
    n_samples=600,
    nx=128,
    nt_out=100,
    t_max=1.0,
    cfl=0.3,
    seed=None,
    bc="copy",       # or "periodic"
    x_min=-1.0,
    x_max=1.0,
):
    if seed is None:
        seed = int(time.time())
    rng = np.random.default_rng(seed)
    print(f"[DATA] Generating {n_samples} samples with seed={seed} ...")

    x_v = np.linspace(x_min, x_max, nx).astype(np.float32)
    t_v = np.linspace(0.0, t_max, nt_out).astype(np.float32)
    X, T = np.meshgrid(x_v, t_v, indexing="ij")  # (nx, nt)

    inputs = np.zeros((n_samples, nx, nt_out, 3), dtype=np.float32)
    targets = np.zeros((n_samples, nx, nt_out), dtype=np.float32)

    start = time.time()
    for i in range(n_samples):
        # random single jump location
        jump = rng.uniform(-0.5, 0.5)

        # random left/right states
        uL = rng.uniform(0.1, 0.9)
        uR = rng.uniform(0.1, 0.9)
        while abs(uL - uR) < 0.05:
            uR = rng.uniform(0.1, 0.9)

        u0 = np.where(x_v < jump, uL, uR).astype(np.float32)

        # Godunov solve: returns u_hist (nt, nx)
        _, _, u_hist = solve_fvm(
            u0,
            nt_out=nt_out,
            x_min=x_min,
            x_max=x_max,
            t_max=t_max,
            cfl=cfl,
            bc=bc,
        )

        # Convert to (nx, nt)
        u_xt = u_hist.T  # (nx, nt)

        # Build input features (nx, nt, 3): [u0(x) repeated in time, X, T]
        u0_rep = np.repeat(u0[:, None], nt_out, axis=1)  # (nx, nt)
        inputs[i] = np.stack([u0_rep, X, T], axis=-1)
        targets[i] = u_xt

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_samples} generated ({(time.time()-start)/60:.2f} min)")

    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float(), x_v, t_v

def build_fno_input(u0: np.ndarray, x_v: np.ndarray, t_v: np.ndarray) -> torch.Tensor:
    """
    Build input of shape (1, nx, nt, 3) = [u0(x) repeated over t, X, T]
    matching your training format.
    """
    nx = len(x_v)
    nt = len(t_v)

    X, T = np.meshgrid(x_v.astype(np.float32), t_v.astype(np.float32), indexing="ij")  # (nx, nt)
    u0 = u0.astype(np.float32)
    u0_rep = np.repeat(u0[:, None], nt, axis=1)  # (nx, nt)

    inp = np.stack([u0_rep, X, T], axis=-1)      # (nx, nt, 3)
    return torch.from_numpy(inp).float().unsqueeze(0)  # (1, nx, nt, 3)


# ----------------------------
# 4) Train + plot
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- settings ---
    n_samples = 600
    nx = 128
    nt_out = 100
    t_max = 1.0
    cfl = 0.3
    epochs = 300
    lr = 1e-3

    # fresh data each run
    x_data, y_data, x_v, t_v = generate_pde_data(
        n_samples=n_samples,
        nx=nx,
        nt_out=nt_out,
        t_max=t_max,
        cfl=cfl,
        seed=None,     # None => different each run
        bc="copy",     # or "periodic"
    )

    loader = DataLoader(
        TensorDataset(x_data, y_data),
        batch_size=20,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = FNO_SpaceTime_Final(modes1=32, modes2=32, width=96, layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print("[TRAIN] starting ...")
    start_time = time.time()
    for epoch in range(epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)

            # MSE (+ tiny L1 sometimes helps shocks)
            loss = F.mse_loss(pred, yb) + 1e-3 * F.l1_loss(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % 50 == 0:
            avg = total_loss / len(loader)
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:03d} | loss={avg:.6e} | lr={lr_now:.2e}")

    print(f"[TRAIN] done in {(time.time()-start_time)/60:.2f} minutes.")

    # --- NEW test initial condition (not from training tensor) ---
    rng = np.random.default_rng(123)

    jump = rng.uniform(-0.5, 0.5)
    uL = rng.uniform(0.1, 0.9)
    uR = rng.uniform(0.1, 0.9)
    while abs(uL - uR) < 0.05:
        uR = rng.uniform(0.1, 0.9)

    u0_new = np.where(x_v < jump, uL, uR).astype(np.float32)
    print(f"[NEW TEST IC] jump={jump:.3f}, uL={uL:.3f}, uR={uR:.3f}")

    # Godunov truth for this NEW IC
    _, _, u_hist_new = solve_fvm(
        u0_new,
        nt_out=nt_out,
        x_min=-1.0,
        x_max=1.0,
        t_max=t_max,
        cfl=cfl,
        bc="copy",
    )
    truth = u_hist_new.T  # (nx, nt)

    # FNO prediction for this NEW IC
    model.eval()
    with torch.no_grad():
        test_in = build_fno_input(u0_new, x_v, t_v).to(device)    # (1, nx, nt, 3)
        pred = model(test_in).cpu().numpy()[0]                    # (nx, nt)

    # --- plots: x horizontal, t vertical ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dat = [pred, truth, np.abs(truth - pred)]
    titles = ["FNO Prediction", "Godunov Truth", "Absolute Error"]
    cmaps = ["jet", "jet", "magma"]

    for i in range(3):
        im = axes[i].pcolormesh(
            x_v, t_v, dat[i].T,  # (nt, nx)
            shading="auto",
            cmap=cmaps[i],
        )
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("t")
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig("flux_u1-u_singlejump.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()