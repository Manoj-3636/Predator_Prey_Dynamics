"""M6: nullclines and normalised direction fields (explicit nested loops)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import P, rm, x_star, y_star

def prey_nullcline(x, p, K): return p['r']/p['a']*(1 - x/K)*(1 + p['a']*p['h']*x)
def pred_nullcline_x(p): return x_star(p)

def grid_field(xmax, ymax, n, p, K, model=rm):
    xs = np.linspace(0.02, xmax, n); ys = np.linspace(0.02, ymax, n)
    U = np.zeros((n, n)); V = np.zeros((n, n)); S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            xd, yd = model(0, [xs[j], ys[i]], p, K)
            norm = np.hypot(xd, yd)
            if abs(xd) > 1e-12: S[i, j] = yd/xd          # slope dy/dx (undefined/vertical if xdot=0)
            U[i, j], V[i, j] = (xd/norm, yd/norm) if norm > 0 else (0, 0)   # unit arrows
    return xs, ys, U, V, S

def plot_field(ax, K, xmax=None, ymax=None, n=22, p=P):
    xmax = xmax or K*1.05; ymax = ymax or 2.6
    xs, ys, U, V, _ = grid_field(xmax, ymax, n, p, K)
    ax.quiver(xs, ys, U, V[:, :], color="0.55", angles="xy", pivot="mid", scale=35, width=0.003)
    xx = np.linspace(0, K, 400); ax.plot(xx, prey_nullcline(xx, p, K), "b-", label="prey nullcline")
    ax.axvline(x_star(p), color="r", label="predator nullcline"); ax.plot(x_star(p), y_star(p, K), "ko")
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.set_xlabel("x (prey)"); ax.set_ylabel("y (predator)")

def make_figs(outdir="../outputs"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a_, (nm, K) in zip(ax, P['K_test'].items()):
        plot_field(a_, K); a_.set_title(f"Vector field + nullclines, K={K:.3f}"); a_.legend(loc="upper right", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{outdir}/fig_7_nullclines_vector_field.png", dpi=300); plt.close()

if __name__ == "__main__":
    for nm, K in P['K_test'].items():
        peak = (K - 1/(P['a']*P['h']))/2
        print(f"K={K:.4f}: nullcline hump peak at x={peak:.4f}, x*={x_star():.4f}, x*<peak? {x_star()<peak}")
    xs, ys, U, V, S = grid_field(3, 2.5, 5, P, 3.0); print("sample slope grid (5x5):\n", np.round(S, 3))
    make_figs()
