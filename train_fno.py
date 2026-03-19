import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
from gudonov import solve_fvm

# --- 1. Spectral Convolution 2D ---
# Follows Definition 3: Convolution via linear transform in Fourier domain
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.complex(torch.randn(in_channels, out_channels, self.modes1, self.modes2), torch.randn(in_channels, out_channels, self.modes1, self.modes2)))
        self.weights2 = nn.Parameter(scale * torch.complex(torch.randn(in_channels, out_channels, self.modes1, self.modes2), torch.randn(in_channels, out_channels, self.modes1, self.modes2)))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, x.shape[1], x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        # Multiply low-frequency modes by learnable weights R
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

# --- 2. Multi-Layer FNO Architecture ---
# Stacks 4 Fourier Layers as specified in Section 5
class FNO_SpaceTime_Final(nn.Module):
    def __init__(self, modes1=24, modes2=24, width=64):
        super(FNO_SpaceTime_Final, self).__init__()
        self.width = width
        # Lifting P: [u0, x, t] -> width
        self.fc0 = nn.Linear(3, self.width) 

        self.convs = nn.ModuleList([SpectralConv2d(self.width, self.width, modes1, modes2) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(4)]) 
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(4)]) 

        # Projection Q
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0, 3, 1, 2)
        for conv, w, bn in zip(self.convs, self.ws, self.bns):
            x = bn(F.gelu(conv(x) + w(x))) # Non-linear activation GELU
        
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

# --- 3. Data Generation (Updated for Concave Flux u(1-u)) ---
# --- 3. Data Generation (Updated for Single Jump / Shock) ---
# --- Updated Import ---
from gudonov import solve_fvm  # Ensure the file is named gudonov.py

# --- 3. Data Generation (Aligned with your Godunov Solver) ---
def generate_pde_data(n_samples=600, nx=128, nt_out=100):
    print(f"Generating {n_samples} Single Shock/Rarefaction samples...")
    inputs, targets = [], []
    
    # Define domain
    x_min, x_max = -1.0, 1.0
    t_max = 1.0
    x_v = np.linspace(x_min, x_max, nx)
    t_v = np.linspace(0, t_max, nt_out)
    
    # Create coordinate grids for FNO input [u0, x, t]
    X, T = np.meshgrid(x_v, t_v, indexing='ij') # Shape: (nx, nt_out)

    for _ in range(n_samples):
        # 1. Random values for jump
        u_left = np.random.uniform(0.1, 0.9)
        u_right = np.random.uniform(0.1, 0.9)
        jump_loc = np.random.uniform(-0.5, 0.5)
        
        # 2. Initial Condition
        u0 = np.where(x_v < jump_loc, u_left, u_right)
        
        # 3. Call your specific solver
        # Returns: x (nx,), t (nt_out,), u_hist (nt_out, nx)
        _, _, u_hist = solve_fvm(u0, nt_out=nt_out, x_min=x_min, x_max=x_max, t_max=t_max, cfl=0.3)
        
        if u_hist is not None:
            # u_hist is (time, space), we need (space, time) to match X and T grids
            target_field = u_hist.T 
            
            # Prepare u0 feature: repeat u0 for all time steps
            u0_rep = np.repeat(u0[:, np.newaxis], nt_out, axis=1)
            
            # Stack features: [nx, nt_out, 3] -> (u0, x, t)
            inputs.append(np.stack([u0_rep, X, T], axis=-1))
            targets.append(target_field) 
            
    return torch.tensor(np.array(inputs), dtype=torch.float32), \
           torch.tensor(np.array(targets), dtype=torch.float32)


# --- 4. Main Training Loop with Analytics ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_data, y_data = generate_pde_data(n_samples=600)
    
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=20, shuffle=True)
    model = FNO_SpaceTime_Final(modes1=24, modes2=24, width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Scheduler: Halves the learning rate every 100 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    start_time = time.time()
    print(f"Training on {device}...")
    
    for epoch in range(301):
        model.train()
        total_mse = 0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(x_b)
            loss = F.mse_loss(out, y_b)
            loss.backward()
            optimizer.step()
            total_mse += loss.item()
        
        scheduler.step()
        if epoch % 50 == 0:
            avg_mse = total_mse / len(loader)
            print(f"Epoch {epoch} | MSE: {avg_mse:.8f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    total_time = time.time() - start_time
    print(f"Training Complete in {total_time/60:.2f} minutes.")

    # --- 5. Visualization & Statistics ---
    # --- 5. Visualization & Statistics (Axes Swapped) ---
    model.eval()
    with torch.no_grad():
        # test_out and pred are shape (nx, nt)
        test_in, test_out = x_data[0:1].to(device), y_data[0].cpu().numpy()
        pred = model(test_in).cpu().numpy()[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dat = [pred, test_out, np.abs(test_out - pred)]
    titles = ["FNO Prediction", "Godunov Truth", "Absolute Error"]

    # Define the actual coordinate ranges
    x_range = np.linspace(-1, 1, 128)
    t_range = np.linspace(0, 1, 100)

    for i in range(3):
        # SWAP HERE: t_range is now Y (vertical), x_range is now X (horizontal)
        # We transpose the data (.T) to match (time, space) shape for the plot
        im = axes[i].pcolormesh(x_range, t_range, dat[i].T, shading='auto', 
                                cmap='jet' if i < 2 else 'magma')
        
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Space (x)")
        axes[i].set_ylabel("Time (t)")
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig("swapped_axes_plot.png")
    plt.show()