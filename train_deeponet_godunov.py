from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Union
from gudonov import solve_fvm
# ============================================================
# 1) GODUNOV FVM SOLVER
# ============================================================


# ============================================================
# 2) DEEPONET ARCHITECTURE
# ============================================================
def make_activation(act: Union[str, nn.Module]) -> nn.Module:
    if isinstance(act, nn.Module): return act
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(act.lower(), nn.Tanh())

def _build_mlp(in_dim, hidden_width, layers, out_dim, activation):
    modules = []
    dim = in_dim
    for _ in range(layers - 1):
        modules.extend([nn.Linear(dim, hidden_width), make_activation(activation)])
        dim = hidden_width
    modules.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*modules)

class DeepONet(nn.Module):
    def __init__(self, branch_in, trunk_in=2, hidden_width=64, branch_layers=3, 
                 trunk_layers=3, latent_dim=64, activation="tanh"):
        super().__init__()
        self.branch = _build_mlp(branch_in, hidden_width, branch_layers, latent_dim, activation)
        self.trunk = _build_mlp(trunk_in, hidden_width, trunk_layers, latent_dim, activation)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_in, trunk_in):
        if branch_in.dim() == 1: branch_in = branch_in.unsqueeze(0)
        b = self.branch(branch_in) 
        t = self.trunk(trunk_in)   
        out = torch.matmul(b, t.T) + self.bias
        # Force output to [0, 1] for physical consistency in LWR model
        return torch.sigmoid(out)

# ============================================================
# 3) VISUALIZATION FUNCTION
# ============================================================
def plot_pred_truth_err(x, t, u_pred, u_true, out_png="deeponet_step_0.7_0.5.png"):
    err = np.abs(u_pred - u_true)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    extent = [x[0], x[-1], t[0], t[-1]]

    # Truth
    im0 = axs[0].imshow(u_true, aspect="auto", origin="lower", extent=extent, cmap="jet")
    axs[0].set_title("Godunov Truth")
    fig.colorbar(im0, ax=axs[0])

    # Prediction
    im1 = axs[1].imshow(u_pred, aspect="auto", origin="lower", extent=extent, cmap="jet")
    axs[1].set_title("DeepONet Prediction")
    fig.colorbar(im1, ax=axs[1])

    # Error
    im2 = axs[2].imshow(err, aspect="auto", origin="lower", extent=extent, cmap="magma")
    axs[2].set_title("|Pred - Truth|")
    fig.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    plt.savefig(out_png, dpi=300)
    print(f"Plot saved to {out_png}")
    plt.show()

# ============================================================
# 4) MAIN TRAINING AND EVALUATION
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nx, nt = 128, 100
    n_train_samples = 300
    
    # Generate Training Data
    x_coords = np.linspace(-1.0, 1.0, nx)
    t_coords = np.linspace(0.0, 1.0, nt)
    
    ICs, Us = [], []
    print(f"Generating {n_train_samples} samples...")
    for _ in range(n_train_samples):
        x0 = np.random.uniform(-0.5, 0.5)
        uL, uR = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
        u0 = np.where(x_coords < x0, uL, uR)
        _, _, u_sol = solve_fvm(u0, nt_out=nt)
        ICs.append(u0)
        Us.append(u_sol)
    
    ics_torch = torch.tensor(np.array(ICs), dtype=torch.float32).to(device)
    # Target shape: (Batch, nx * nt)
    targets_torch = torch.tensor(np.array(Us).transpose(0, 2, 1).reshape(n_train_samples, -1), dtype=torch.float32).to(device)

    # Trunk input: (nx * nt, 2)
    X_grid, T_grid = np.meshgrid(x_coords, t_coords, indexing='ij')
    trunk_in = torch.tensor(np.stack([X_grid.ravel(), T_grid.ravel()], axis=1), dtype=torch.float32).to(device)

    # Model Setup
    model = DeepONet(branch_in=nx, hidden_width=128, branch_layers=4, trunk_layers=4, latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")
    for step in range(1501):
        model.train()
        optimizer.zero_grad()
        pred = model(ics_torch, trunk_in)
        loss = nn.MSELoss()(pred, targets_torch)
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            print(f"Step {step:4d} | MSE: {loss.item():.2e}")

    # Final Test Case: Single jump 0.7 -> 0.2 at x=0
    u0_test = np.where(x_coords < 0.0, 0.5, 0.7)
    x_ref, t_ref, u_true = solve_fvm(u0_test, nt_out=nt)
    
    model.eval()
    with torch.no_grad():
        test_ic_torch = torch.tensor(u0_test, dtype=torch.float32).to(device)
        u_pred = model(test_ic_torch, trunk_in).cpu().numpy().reshape(nx, nt).T
    
    plot_pred_truth_err(x_ref, t_ref, u_pred, u_true, out_png="deeponet_step_0.5_0.7.png")

if __name__ == "__main__":
    main()