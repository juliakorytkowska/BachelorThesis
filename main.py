# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from gudonov import solve_fvm

def shock_ridge(u_hist: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0]
    dudx = np.gradient(u_hist, dx, axis=1)          # (nt, nx)
    idx = np.argmax(np.abs(dudx), axis=1)           # (nt,)
    return x[idx]

def main():
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    savepath = os.path.join(outdir, "godunov_copyBC_riemann.png")
    print("Saving to:", os.path.abspath(savepath))

    nx = 600
    nt_out = 220
    x = np.linspace(-1.0, 1.0, nx)

    # For concave flux f(u)=u(1-u):
    # uL < uR gives a SHOCK
    uL, uR = 0.2, 0.8
    u0 = np.where(x < 0.0, uL, uR)

    x, t, u_hist = solve_fvm(
        u0=u0,
        nt_out=nt_out,
        x_min=-1.0,
        x_max=1.0,
        t_max=4.0,
        cfl=0.35,
        bc="copy",            # <--- ghost cells copy boundary values (not zero)
    )
    print("u_hist[0] min/max:", u_hist[0].min(), u_hist[0].max())
    print("u_hist[-1] min/max:", u_hist[-1].min(), u_hist[-1].max())
    print("Linf change:", np.max(np.abs(u_hist[-1] - u_hist[0])))

    x_shock = shock_ridge(u_hist, x)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(x, t, u_hist, shading="auto")
    ax.plot(x_shock, t, "w-", linewidth=2, label=r"argmax$_x |u_x|$")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(r"Godunov: $u_t + (u(1-u))_x = 0$ (Riemann IC, copy ghost cells)")
    fig.colorbar(im, ax=ax, label="u(x,t)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()