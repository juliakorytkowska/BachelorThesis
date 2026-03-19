# train_low_predict_high.py
# ============================================================
# Train on low resolution (nx_low, nt_low),
# Predict on high resolution (nx_high, nt_high)
# for u_t + (u(1-u))_x = 0 using your Godunov solver + FNO.
# ============================================================

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from gudonov import solve_fvm  # your file

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
class FNO_SpaceTime(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64, layers=4):
        super().__init__()
        self.fc0 = nn.Linear(3, width)

        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(layers)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, nx, nt, 3)
        x = self.fc0(x)            # (B, nx, nt, width)
        x = x.permute(0, 3, 1, 2)  # (B, width, nx, nt)

        for conv, w, bn in zip(self.convs, self.ws, self.bns):
            x = bn(conv(x) + w(x))
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)  # (B, nx, nt, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)  # (B, nx, nt)
        return x


# ----------------------------
# Helpers: random single-jump IC
# ----------------------------
def sample_single_jump_ic(x, rng):
    jump = rng.uniform(-0.5, 0.5)
    uL = rng.uniform(0.1, 0.9)
    uR = rng.uniform(0.1, 0.9)
    while abs(uL - uR) < 0.05:
        uR = rng.uniform(0.1, 0.9)
    u0 = np.where(x < jump, uL, uR).astype(np.float32)
    return u0, jump, uL, uR


# ----------------------------
# 3) Data generation at a given resolution
# ----------------------------
def generate_dataset(n_samples, nx, nt_out, t_max, cfl, bc, x_min=-1.0, x_max=1.0, seed=0):
    rng = np.random.default_rng(seed)

    x = np.linspace(x_min, x_max, nx).astype(np.float32)
    t = np.linspace(0.0, t_max, nt_out).astype(np.float32)
    X, T = np.meshgrid(x, t, indexing="ij")  # (nx, nt)

    inputs = np.zeros((n_samples, nx, nt_out, 3), dtype=np.float32)
    targets = np.zeros((n_samples, nx, nt_out), dtype=np.float32)

    for i in range(n_samples):
        u0, *_ = sample_single_jump_ic(x, rng)

        _, _, u_hist = solve_fvm(
            u0,
            nt_out=nt_out,
            x_min=x_min,
            x_max=x_max,
            t_max=t_max,
            cfl=cfl,
            bc=bc,
        )

        u_xt = u_hist.T  # (nx, nt)

        u0_rep = np.repeat(u0[:, None], nt_out, axis=1)
        inputs[i] = np.stack([u0_rep, X, T], axis=-1)
        targets[i] = u_xt

    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float(), x, t


# ----------------------------
# 4) Build input tensor (u0,x,t) for ANY resolution
# ----------------------------
def build_input_from_u0(u0, x, t):
    # u0: (nx,), x: (nx,), t: (nt,)
    X, T = np.meshgrid(x.astype(np.float32), t.astype(np.float32), indexing="ij")
    u0_rep = np.repeat(u0[:, None], len(t), axis=1).astype(np.float32)
    inp = np.stack([u0_rep, X, T], axis=-1)  # (nx, nt, 3)
    return torch.from_numpy(inp).float()
def field_mse(u_pred, u_true):
    return float(np.mean((u_pred - u_true) ** 2))

# ----------------------------
# 5) Train low-res, evaluate high-res
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------------------
    # LOW-RES TRAINING SETTINGS
    # ----------------------------
    n_train = 600
    nx_low = 128
    nt_low = 128
    t_max = 1.0
    cfl = 0.3
    bc = "copy"

    batch_size = 20 # number of training examples processed together before the model updates its weights once
    epochs = 300
    lr = 1e-3

    # ----------------------------
    # HIGH-RES TEST SETTINGS
    # ----------------------------
    nx_high = 256
    nt_high = 256

    # Generate low-res training data
    print(f"[DATA] train on low-res: nx={nx_low}, nt={nt_low}")
    x_train, y_train, x_low, t_low = generate_dataset(
        n_samples=n_train,
        nx=nx_low,
        nt_out=nt_low,
        t_max=t_max,
        cfl=cfl,
        bc=bc,
        seed=0
    )

    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=batch_size, shuffle=True)

    # Model: modes must be <= nx_low, nt_low (safe pick)
    model = FNO_SpaceTime(modes1=16, modes2=16, width=64, layers=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    # Train
    print("[TRAIN] ...")
    t0 = time.time()
    for ep in range(epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb) + 1e-3 * F.l1_loss(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()

        sch.step()
        if ep % 50 == 0:
            print(f"Epoch {ep:03d} | loss={total/len(loader):.3e} | lr={sch.get_last_lr()[0]:.2e}")

    print(f"[TRAIN] done in {(time.time()-t0)/60:.2f} min")

    # ----------------------------
    # Create a NEW high-res test IC + high-res Godunov truth
    # ----------------------------
    rng = np.random.default_rng(123)
    xh = np.linspace(-1.0, 1.0, nx_high).astype(np.float32)
    th = np.linspace(0.0, t_max, nt_high).astype(np.float32)

    u0_high, jump, uL, uR = sample_single_jump_ic(xh, rng)
    print(f"[TEST IC] jump={jump:.3f}, uL={uL:.3f}, uR={uR:.3f}")

    # High-res Godunov truth
    _, _, u_hist_high = solve_fvm(
        u0_high,
        nt_out=nt_high,
        x_min=-1.0,
        x_max=1.0,
        t_max=t_max,
        cfl=cfl,
        bc=bc,
    )
    truth_high = u_hist_high.T  # (nx_high, nt_high)

    # ----------------------------
    # Evaluate SAME MODEL on high-res grid
    # ----------------------------
    model.eval()
    with torch.no_grad():
        inp_high = build_input_from_u0(u0_high, xh, th)  # (nx_high, nt_high, 3)
        pred_high = model(inp_high[None, ...].to(device)).cpu().numpy()[0]  # (nx_high, nt_high)

    # MSE over the whole space-time grid
    mse_high = np.mean((truth_high - pred_high) ** 2)
    print(f"[EVAL] MSE on HIGH-res = {mse_high:.3e}")

    # Relative L2 error
    rel_l2 = np.linalg.norm(truth_high - pred_high) / (np.linalg.norm(truth_high) + 1e-12)
    print(f"[EVAL] relL2 on HIGH-res = {rel_l2:.4%}")

    # ----------------------------
    # Plot on HIGH-res grid
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    data = [pred_high, truth_high, np.abs(truth_high - pred_high)]
    titles = ["FNO Prediction (trained low-res)", "Godunov Truth (high-res)", "Absolute Error (high-res)"]
    cmaps = ["jet", "jet", "magma"]

    for i in range(3):
        im = axes[i].pcolormesh(xh, th, data[i].T, shading="auto", cmap=cmaps[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("t")
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig("low_train_high_test_2.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()