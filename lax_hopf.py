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
    def __init__(self, modes1=24, modes2=24, width=64):
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



def solve_lax_hopf(u0_values, nx=128, nt_out=100, x_min=-1.0, x_max=1.0, t_max=1.0):
    x_grid = np.linspace(x_min, x_max, nx)
    t_grid = np.linspace(0, t_max, nt_out)
    u_hist = np.zeros((nt_out, nx))
    
    # 1. Precompute U0 (Integral of u0)
    dx = x_grid[1] - x_grid[0]
    U0 = np.cumsum(u0_values) * dx 

    def get_U0(y):
        # Interpolate the integrated IC to find value at any y
        return np.interp(y, x_grid, U0)

    # 2. Solve Hopf-Lax for each (x, t)
    for i, t in enumerate(t_grid):
        if t == 0:
            u_hist[i, :] = u0_values
            continue
        
        for j, x in enumerate(x_grid):
            # For f(u) = u - u^2, the conjugate is H*(v) = 0.25 * (1 - v)^2
            # Formula: u(x,t) = argmax_y [ U0(y) + t * H*((x-y)/t) ]
            def objective(y):
                v = (x - y) / t
                h_star = 0.25 * (1 - v)**2
                return -(get_U0(y) + t * h_star) # Negative for maximization

            # Find the optimal characteristic y
            res = minimize_scalar(objective, bounds=(x_min, x_max), method='bounded')
            
            # The solution u(x,t) is the value of u0 at the source y
            # This handles shocks perfectly without numerical diffusion
            u_hist[i, j] = np.interp(res.x, x_grid, u0_values)
            
    return x_grid, t_grid, u_hist


# --- 3. Data Generation (Lax-Hopf Staircase) ---
def generate_pde_data(n_samples=200, nx=128, nt_out=100):
    # Reduced n_samples because Lax-Hopf is slow to compute exactly
    print(f"Generating {n_samples} Exact Lax-Hopf Staircase samples...")
    inputs, targets = [], []
    x_v, t_v = np.linspace(-1, 1, nx), np.linspace(0, 1.0, nt_out)
    X, T = np.meshgrid(x_v, t_v, indexing='ij')

    for _ in range(n_samples):
        # Create Staircase: High values -> Low values (for shocks)
        v1 = np.random.uniform(0.8, 0.9)
        v2 = np.random.uniform(0.6, 0.7)
        v3 = np.random.uniform(0.4, 0.5)
        v4 = np.random.uniform(0.1, 0.2)
        v = [v1, v2, v3, v4]
        j1 = np.random.uniform(-0.8, -0.5)
        j2 = np.random.uniform(-0.2, 0.0)
        j3 = np.random.uniform(0.3, 0.6)
        jumps = [j1, j2, j3]
        
        u0 = np.full(nx, v[0])
        for i in range(3):
            u0[x_v >= jumps[i]] = v[i+1]
        
        # Call the new Lax-Hopf solver
        _, _, u_hist = solve_lax_hopf(u0, nx=nx, nt_out=nt_out)
        
        u0_rep = np.repeat(u0[:, np.newaxis], nt_out, axis=1)
        inputs.append(np.stack([u0_rep, X, T], axis=-1))
        targets.append(u_hist.T) 
            
    return torch.tensor(np.array(inputs), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)

# --- 4. Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_data, y_data = generate_pde_data(n_samples=100) # Use smaller batch for demonstration
    
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=10, shuffle=True)
    model = FNO_SpaceTime_Final(modes1=24, modes2=24, width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    print("Training on Lax-Hopf Exact Ground Truth...")
    for epoch in range(301):
        model.train()
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x_b), y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | MSE: {loss.item():.8f}")

    # Visualization
    model.eval()
    with torch.no_grad():
        test_in, test_out = x_data[0:1].to(device), y_data[0].cpu().numpy()
        pred = model(test_in).cpu().numpy()[0]
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dat = [pred, test_out, np.abs(test_out - pred)]
    titles = ["FNO (Lax-Hopf Trained)", "Exact Lax-Hopf Truth", "Absolute Error"]
    for i in range(3):
        im = axes[i].pcolormesh(np.linspace(0, 1, 100), np.linspace(-1, 1, 128), dat[i], shading='auto', cmap='jet' if i<2 else 'magma')
        axes[i].set_title(titles[i]); plt.colorbar(im, ax=axes[i])
    plt.tight_layout(); plt.savefig("lax_hopf_fno 3.png"); plt.show()