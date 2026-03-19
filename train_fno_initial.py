import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from solver import solve_burgers_fvm

# --- 1. Spectral Convolution 2D ---
# Follows Definition 3: Convolution instantiated via linear transform in Fourier domain [cite: 172]
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
        # Multiply relevant Fourier modes by learnable weights R [cite: 184, 185]
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

# --- 2. Multi-Layer FNO Architecture ---
# Stacks 4 Fourier Layers as specified in Section 5 
class FNO_SpaceTime_Improved(nn.Module):
    def __init__(self, modes1=20, modes2=20, width=64):
        super(FNO_SpaceTime_Improved, self).__init__()
        self.width = width
        # Lifting P: projects [u0, x, t] to higher dimension [cite: 124, 140]
        self.fc0 = nn.Linear(3, self.width) 

        # 4 Iterative updates as per Figure 2 [cite: 124, 129]
        self.convs = nn.ModuleList([SpectralConv2d(self.width, self.width, modes1, modes2) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(4)]) # Local linear W [cite: 147]
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(4)]) # Batchnorm 

        # Projection Q: Two-layer MLP back to target dimension [cite: 125, 142]
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0, 3, 1, 2)
        for conv, w, bn in zip(self.convs, self.ws, self.bns):
            # Combine non-local spectral integral and local activation [cite: 85, 145]
            x = bn(F.gelu(conv(x) + w(x)))
        
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

# --- 3. Data Generation ---
def generate_pw_data(n_samples=600, nx=128, nt_out=100):
    print(f"Generating {n_samples} Piecewise Constant trajectories...")
    inputs, targets = [], []
    x_v, t_v = np.linspace(-1, 1, nx), np.linspace(0, 1.0, nt_out)
    X, T = np.meshgrid(x_v, t_v, indexing='ij')

    for _ in range(n_samples):
        jump = np.random.uniform(-0.5, 0.5)
        u0 = np.where(x_v < jump, np.random.uniform(0.6, 0.9), np.random.uniform(0.3, 0.5))
        _, _, u_hist = solve_burgers_fvm(u0, nx=nx, nt_out=nt_out, cfl=0.9)
        u0_rep = np.repeat(u0[:, np.newaxis], nt_out, axis=1)
        inputs.append(np.stack([u0_rep, X, T], axis=-1))
        targets.append(u_hist.T) 
        
    return torch.tensor(np.array(inputs), dtype=torch.float32), \
           torch.tensor(np.array(targets), dtype=torch.float32)

# --- 4. Main training and Visuals ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_data, y_data = generate_pw_data(n_samples=600)
    
    # Mini-batching to avoid OutOfMemoryError 
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=20, shuffle=True)
    model = FNO_SpaceTime_Improved(modes1=20, modes2=20, width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training on {device} with mini-batches...")
    for epoch in range(301):
        total_loss = 0
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Avg Loss: {total_loss/len(loader):.6f}")

    # Final Prediction Visual
    model.eval()
    with torch.no_grad():
        test_in, test_out = x_data[0:1].to(device), y_data[0].numpy()
        pred = model(test_in).cpu().numpy()[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["FNO Prediction", "Godunov Truth", "Absolute Error"]
    dat = [pred, test_out, np.abs(test_out - pred)]
    
    for i in range(3):
        im = axes[i].pcolormesh(np.linspace(0, 1, 100), np.linspace(-1, 1, 128), dat[i], cmap='jet' if i<2 else 'magma')
        axes[i].set_title(titles[i]); plt.colorbar(im, ax=axes[i])
    plt.tight_layout(); plt.savefig("final_fno_report.png"); plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
from solver import solve_burgers_fvm

# --- 1. Spectral Convolution 2D ---
# Follows Definition 3: Convolution instantiated via linear transform in Fourier domain [cite: 172]
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        '''modes1, modes2 define the number of low-frequency Fourier modes to keep'''
        super(SpectralConv2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        #weights are what the model learns during training to effectively solve the PDE
        self.weights1 = nn.Parameter(scale * torch.complex(torch.randn(in_channels, out_channels, self.modes1, self.modes2), torch.randn(in_channels, out_channels, self.modes1, self.modes2)))
        self.weights2 = nn.Parameter(scale * torch.complex(torch.randn(in_channels, out_channels, self.modes1, self.modes2), torch.randn(in_channels, out_channels, self.modes1, self.modes2)))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, x.shape[1], x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        # Multiply relevant Fourier modes by learnable weights R 
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

# --- 2. Multi-Layer FNO Architecture ---
# Stacks 4 Fourier Layers as specified in Section 5 
class FNO_SpaceTime_Improved(nn.Module):
    def __init__(self, modes1=20, modes2=20, width=64):
        super(FNO_SpaceTime_Improved, self).__init__()
        self.width = width
        '''takes 3 inputs for every point in the grid: IC, x, t and lifts to a higher dim (defined by width 64)'''
        # Lifting P: projects [u0, x, t] to higher dimension 
        self.fc0 = nn.Linear(3, self.width) 
        ''' operator learning , 4 layers; uses Fourier tarsnform to perform operation'''
        # 4 Iterative updates
        self.convs = nn.ModuleList([SpectralConv2d(self.width, self.width, modes1, modes2) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(4)]) # Local linear W [cite: 147]
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(4)]) # Batchnorm 

        # Projection Q: Two-layer MLP back to target dimension 
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        '''engine: defines how input data is transformed to the solution by moving through neural-operator's layers'''
        x = self.fc0(x).permute(0, 3, 1, 2)
        for conv, w, bn in zip(self.convs, self.ws, self.bns):
            # Combine non-local spectral integral and local activation
            x = bn(F.gelu(conv(x) + w(x)))
        
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


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
'''creating the synthetic dataset that the FNO uses to "learn" the physics of the PDE. 
Instead of a formula, the model sees 600 examples of how an initial state evolves over time.'''
def generate_pde_data(n_samples=600, nx=128, nt_out=100):
    print(f"Generating {n_samples} Piecewise Constant samples for f(u)=u(1-u)...")
    inputs, targets = [], []
    x_v, t_v = np.linspace(-1, 1, nx), np.linspace(0, 1.0, nt_out)
    X, T = np.meshgrid(x_v, t_v, indexing='ij')

    for _ in range(n_samples):
        jump = np.random.uniform(-0.5, 0.5)
        # Input restricted to [0.1, 0.9] for traffic/concentration physics
        u0 = np.where(x_v < jump, np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
        _, _, u_hist = solve_burgers_fvm(u0, nx=nx, nt_out=nt_out, cfl=0.9)
        u0_rep = np.repeat(u0[:, np.newaxis], nt_out, axis=1)
        inputs.append(np.stack([u0_rep, X, T], axis=-1))
        targets.append(u_hist.T) 
        
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
    model.eval()
    with torch.no_grad():
        test_in, test_out = x_data[0:1].to(device), y_data[0].cpu().numpy()
        pred = model(test_in).cpu().numpy()[0]
        
    # Relative L2 Error calculation
    rel_l2 = np.linalg.norm(test_out - pred) / np.linalg.norm(test_out)
    print(f"Final Relative L2 Error: {rel_l2:.4%}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dat = [pred, test_out, np.abs(test_out - pred)]
    titles = ["FNO Prediction", "Godunov Truth", "Absolute Error"]
    
    for i in range(3):
        im = axes[i].pcolormesh(np.linspace(0, 1, 100), np.linspace(-1, 1, 128), dat[i], shading='auto', cmap='jet' if i<2 else 'magma')
        axes[i].set_title(titles[i])
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig("analysis_report_flux_u1-u.png")
    plt.show()

#GOOD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
from solver import solve_burgers_fvm

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
def generate_pde_data(n_samples=600, nx=128, nt_out=100):
    print(f"Generating {n_samples} Piecewise Constant samples for f(u)=u(1-u)...")
    inputs, targets = [], []
    x_v, t_v = np.linspace(-1, 1, nx), np.linspace(0, 1.0, nt_out)
    X, T = np.meshgrid(x_v, t_v, indexing='ij')

    for _ in range(n_samples):
        jump = np.random.uniform(-0.5, 0.5)
        # Input restricted to [0.1, 0.9] for traffic/concentration physics
        u0 = np.where(x_v < jump, np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
        _, _, u_hist = solve_burgers_fvm(u0, nx=nx, nt_out=nt_out, cfl=0.9)
        u0_rep = np.repeat(u0[:, np.newaxis], nt_out, axis=1)
        inputs.append(np.stack([u0_rep, X, T], axis=-1))
        targets.append(u_hist.T)
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
model.eval()
with torch.no_grad():
    test_in, test_out = x_data[0:1].to(device), y_data[0].cpu().numpy()
    pred = model(test_in).cpu().numpy()[0]
    # Relative L2 Error calculation
    rel_l2 = np.linalg.norm(test_out - pred) / np.linalg.norm(test_out)
    print(f"Final Relative L2 Error: {rel_l2:.4%}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dat = [pred, test_out, np.abs(test_out - pred)]
titles = ["FNO Prediction", "Godunov Truth", "Absolute Error"]
for i in range(3):
im = axes[i].pcolormesh(np.linspace(0, 1, 100), np.linspace(-1, 1, 128), dat[i], shading='auto', cmap='jet' if i<2 else 'magma')
axes[i].set_title(titles[i])
plt.colorbar(im, ax=axes[i])
plt.tight_layout()
plt.savefig("analysis_report_flux_u1-u 4.png")
plt.show()