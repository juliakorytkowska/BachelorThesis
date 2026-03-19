import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize_scalar

# --- 1. Spectral Convolution 2D ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.complex(torch.randn(in_channels, out_channels, self.modes1, self.modes2), torch.randn(in_channels, out_channels, self.modes1, self.modes2)))
        self.weights2 = nn.Parameter(scale * torch.complex(torch.randn(in_channels, out_channels, self.modes1, self.modes2), torch.randn(in_channels, out_channels, self.modes1, self.modes2)))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, x.shape[1], x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

# --- 2. FNO Architecture ---
class FNO_SpaceTime_Final(nn.Module):
    def __init__(self, modes1=32, modes2=32, width=64):
        super(FNO_SpaceTime_Final, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(3, self.width) 
        self.convs = nn.ModuleList([SpectralConv2d(self.width, self.width, modes1, modes2) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(4)]) 
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(4)]) 
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0, 3, 1, 2)
        for conv, w, bn in zip(self.convs, self.ws, self.bns):
            x = bn(F.gelu(conv(x) + w(x)))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

# --- 3. Exact Lax-Hopf Solver ---
def solve_lax_hopf(u0_values, nx=128, nt_out=100, x_min=-1.0, x_max=1.0, t_max=1.0):
    x_grid = np.linspace(x_min, x_max, nx)
    t_grid = np.linspace(0, t_max, nt_out)
    u_hist = np.zeros((nt_out, nx))
    dx = x_grid[1] - x_grid[0]
    U0 = np.cumsum(u0_values) * dx 

    def get_U0(y):
        return np.interp(y, x_grid, U0)

    for i, t in enumerate(t_grid):
        if t == 0:
            u_hist[i, :] = u0_values
            continue
        for j, x in enumerate(x_grid):
            # Concave flux f(u)=u(1-u) requires maximization
            def objective(y):
                v = (x - y) / t
                h_star = 0.25 * (1 - v)**2
                return -(get_U0(y) + t * h_star) 

            res = minimize_scalar(objective, bounds=(x_min, x_max), method='bounded')
            u_hist[i, j] = np.interp(res.x, x_grid, u0_values)
            
    return u_hist

# --- 4. Complex Data Generation (High Sample Size) ---
def generate_pde_data(n_samples=200, nx=128, nt_out=100):
    print(f"Generating {n_samples} samples at resolution {nx}x{nt_out}...")
    inputs, targets = [], []
    x_v, t_v = np.linspace(-1, 1, nx), np.linspace(0, 1.0, nt_out)
    X, T = np.meshgrid(x_v, t_v, indexing='ij')

    for s in range(n_samples):
        # 5 regions = 4 sharp staircase shocks
        v = sorted(np.random.uniform(0.1, 0.9, 5), reverse=True)
        jumps = sorted(np.random.uniform(-0.8, 0.4, 4))
        u0 = np.full(nx, v[0])
        for i in range(4):
            u0[x_v >= jumps[i]] = v[i+1]
        
        u_hist = solve_lax_hopf(u0, nx=nx, nt_out=nt_out)
        u0_rep = np.repeat(u0[:, np.newaxis], nt_out, axis=1)
        inputs.append(np.stack([u0_rep, X, T], axis=-1))
        targets.append(u_hist.T) 
        if (s+1) % 50 == 0: print(f"Progress: {s+1}/{n_samples}")
            
    return torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)

# --- 5. Main Loop with Training and Mesh Sensitivity ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Training Phase (Learning on Low Resolution)
    train_res = 64
    x_train, y_train = generate_pde_data(n_samples=200, nx=train_res, nt_out=train_res)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=20, shuffle=True)
    
    model = FNO_SpaceTime_Final(modes1=32, modes2=32, width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    print(f"Training complex model on {device}...")
    for epoch in range(301):
        model.train()
        total_loss = 0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x_b), y_b)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 50 == 0: print(f"Epoch {epoch} | MSE: {total_loss/len(loader):.8f}")

    # 2. Mesh Sensitivity Test (Applying to High Resolution)
    print("\n--- Testing Mesh Sensitivity (Zero-Shot) ---")
    for test_res in [64, 128, 256]:
        x_test, y_test = generate_pde_data(n_samples=1, nx=test_res, nt_out=test_res)
        model.eval()
        with torch.no_grad():
            pred = model(x_test.to(device)).cpu().numpy()[0]
            truth = y_test[0].numpy()
            rel_l2 = np.linalg.norm(truth - pred) / np.linalg.norm(truth)
            print(f"Resolution {test_res}x{test_res} | Rel L2 Error: {rel_l2:.4%}")

    # 3. Final Visualization at High Res
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dat = [pred, truth, np.abs(truth - pred)]
    titles = [f"FNO Inference ({test_res}x{test_res})", "Lax-Hopf Ground Truth", "Absolute Error"]
    for i in range(3):
        im = axes[i].pcolormesh(np.linspace(0, 1, test_res), np.linspace(-1, 1, test_res), dat[i], shading='auto', cmap='jet' if i<2 else 'magma')
        axes[i].set_title(titles[i]); plt.colorbar(im, ax=axes[i])
    plt.tight_layout(); plt.savefig("final_mesh_test.png"); plt.show()