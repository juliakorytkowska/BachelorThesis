import os
import numpy as np
import matplotlib.pyplot as plt

from gudonov import solve_fvm


def make_sinusoidal_piecewise(
    x: np.ndarray,
    mean: float = 0.55,
    amp: float = 0.28,
    waves: float = 2,          # number of sine periods across [-1,1]
    n_levels: int = 5,           # quantization levels (more -> closer to sinusoid)
):
    """
    Build a sinusoidal IC and quantize it to piecewise-constant steps.
    Keeps values in [0,1] by clipping.
    """
    # sinusoid on [-1,1] with 'waves' periods
    u = mean + amp * np.sin(2.0 * np.pi * waves * (x - x.min()) / (x.max() - x.min()))
    u = np.clip(u, 0.02, 0.98)

    # quantize into n_levels equally spaced bins
    umin, umax = u.min(), u.max()
    levels = np.linspace(umin, umax, n_levels)
    # map each u to nearest level
    idx = np.abs(u[:, None] - levels[None, :]).argmin(axis=1)
    uq = levels[idx]
    return u, uq


def track_fronts(U, x, t, grad_thresh=8.0):
    """
    Very simple 'front tracker' to draw red lines:
    - compute |du/dx| each time slice
    - keep x positions where |du/dx| exceeds threshold
    This is NOT true wave-front tracking, but produces similar red overlays.
    """
    dx = x[1] - x[0]
    xs_list = []
    ts_list = []
    for n in range(len(t)):
        dudx = np.gradient(U[n], dx)
        mask = np.abs(dudx) > grad_thresh
        xs = x[mask]
        if xs.size == 0:
            continue
        # decimate points so we don't draw a solid red carpet
        xs = xs[:: max(1, xs.size // 80)]
        xs_list.append(xs)
        ts_list.append(np.full_like(xs, t[n]))
    if not xs_list:
        return None, None
    return np.concatenate(xs_list), np.concatenate(ts_list)


def run(save_name="case_sinusoidal_many_fronts.png"):
    os.makedirs("results", exist_ok=True)

    nx = 600
    nt_out = 240
    x_min, x_max = -1.0, 1.0
    t_max = 1.0

    x = np.linspace(x_min, x_max, nx)

    # --- build IC: sinusoid then piecewise-constant steps ---
    u_sin, u0 = make_sinusoidal_piecewise(
        x,
        mean=0.55,
        amp=0.28,
        waves=2,
        n_levels=5,
    )

    # --- solve ---
    xg, tg, U = solve_fvm(
        u0,
        nt_out=nt_out,
        x_min=x_min,
        x_max=x_max,
        t_max=t_max,
        cfl=0.3,
        bc="copy",
    )

    # --- crude front overlay points (red) ---


    # --- plot ---
    fig = plt.figure(figsize=(11, 4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x, u0, lw=2)
    ax1.set_title("Initial condition (piecewise-constant sinusoid)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    pcm = ax2.pcolormesh(xg, tg, U, shading="auto")
    ax2.set_title("Godunov solution u(x,t)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    plt.colorbar(pcm, ax=ax2, label="u")


    fig.suptitle("Sinusoidal IC → many shocks/rarefactions (Godunov)")
    plt.tight_layout()

    outpath = os.path.join("results", save_name)
    plt.savefig(outpath, dpi=300)
    print("Saved to:", outpath)
    plt.show()


if __name__ == "__main__":
    run()