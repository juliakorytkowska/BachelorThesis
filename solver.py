import numpy as np


def godunov_flux(u):
    uL = u[:-1]
    uR = u[1:]

    # New PDE flux: f(u) = u * (1 - u)
    def f(val): return val * (1.0 - val)

    fL = f(uL)
    fR = f(uR)

    # For f(u) = u - u^2, the derivative f'(u) = 1 - 2u.
    # The sonic point (where f'(u) = 0) is at u = 0.5.
    
    # Rarefaction logic
    rare = uL <= uR
    fhat = np.empty_like(uL)
    
    # If uL < uR, we look for the minimum flux in [uL, uR]
    # Since it's a downward parabola, the min is at uL or uR
    fhat[rare] = np.minimum(fL[rare], fR[rare])

    # Shock logic: shock speed s = [f(uR) - f(uL)] / [uR - uL]
    # For this flux, s = (uR - uR^2 - uL + uL^2) / (uR - uL) = 1 - (uR + uL)
    shock = ~rare
    s = 1.0 - (uL + uR)
    fhat[shock] = np.where(s[shock] >= 0.0, fL[shock], fR[shock])
    
    return fhat

def compute_time_step(u, dx, cfl, nu, t, t_max):
    umax = max(1e-6, np.max(np.abs(u)))
    dt_adv = cfl * dx / umax
    dt_diff = np.inf # Purely hyperbolic case
    dt = min(dt_adv, dt_diff, t_max - t)
    return dt

def add_ghost_cells(u):
    u_ext = np.zeros(len(u) + 2)
    u_ext[1:-1] = u
    u_ext[0] = u[0]    # Copy left boundary
    u_ext[-1] = u[-1]  # Copy right boundary
    return u_ext

def solve_burgers_fvm(u_init, nx=128, nt_out=100, x_min=-1.0, x_max=1.0, t_max=1.0, cfl=0.4):
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    u = u_init.copy()
    
    t_out = np.linspace(0.0, t_max, nt_out)
    u_hist = np.zeros((nt_out, nx), dtype=np.float32)
    u_hist[0] = u
    
    t = 0.0
    k = 1
    
    while k < nt_out:
        dt = compute_time_step(u, dx, cfl, 0, t, t_max)
        u_ext = add_ghost_cells(u)
        fhat = godunov_flux(u_ext)
        
        u = u - (dt / dx) * (fhat[1:] - fhat[:-1])
        t += dt
        
        # Capture snapshots for the trajectory
        while k < nt_out and t >= t_out[k] - 1e-12:
            u_hist[k] = u
            k += 1
            
    return x, t_out, u_hist



from scipy.optimize import minimize_scalar

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