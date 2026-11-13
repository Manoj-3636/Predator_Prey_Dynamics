"""M7: phase portraits, limit cycle and detection."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import P, rm, x_star, y_star, K_star
from rk4 import rk4_solve
from fields import plot_field

ICS = [(0.1, 0.1), (0.5, 2.0), (1.0, 0.3), (0.8, 1.5), (0.3, 0.2)]

def portraits(outdir="../outputs"):
    for nm, K in P['K_test'].items():
        fig, ax = plt.subplots(figsize=(6.2, 5.2)); plot_field(ax, K, xmax=min(K*1.05, 3.2), ymax=(5.0 if K > 2 else 2.6))
        for z0 in ICS:
            t, Z = rk4_solve(rm, np.array(z0), 0, 200, 0.01, P, K); ax.plot(Z[:, 0], Z[:, 1], lw=1)
            ax.plot(*z0, "g^", ms=4)
        ax.set_title(f"Phase portrait, K={K:.4f} ({nm})"); ax.legend(loc="upper right", fontsize=7)
        plt.tight_layout(); plt.savefig(f"{outdir}/fig_8_phase_portrait_{nm}.png", dpi=300); plt.close()

def peaks(x):
    i = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1; return x[i]

def limit_cycle(K, T=1500, h=0.01, outdir="../outputs", save=True):
    t, Z = rk4_solve(rm, np.array([0.5, 0.5]), 0, T, h, P, K)
    keep = t > 0.7*T; pk = peaks(Z[keep, 0])
    if save:
        fig, ax = plt.subplots(figsize=(6.2, 5.2)); plot_field(ax, K, xmax=3.2, ymax=5.0)
        ax.plot(Z[:, 0], Z[:, 1], "c", lw=.4, label="transient"); ax.plot(Z[keep, 0], Z[keep, 1], "m", lw=1.4, label="limit cycle")
        ax.set_title(f"Limit cycle, K={K}"); ax.legend(loc="upper right", fontsize=7)
        plt.tight_layout(); plt.savefig(f"{outdir}/fig_8_limit_cycle.png", dpi=300); plt.close()
    return pk, Z[keep]

def onset_scan(Ks, T=4000, h=0.02):
    res = []
    for K in Ks:
        t, Z = rk4_solve(rm, np.array([x_star() + 0.05, y_star(P, K)]), 0, T, h, P, K)
        late = Z[t > 0.8*T, 0]; res.append((K, late.max() - late.min()))
    return res

if __name__ == "__main__":
    portraits()
    pk, Zl = limit_cycle(3.0)
    print("late prey peaks (K=3):", np.round(pk[:5], 5), " spread:", pk.max()-pk.min())
    print(f"late-time amplitude x: [{Zl[:,0].min():.4f}, {Zl[:,0].max():.4f}]  y: [{Zl[:,1].min():.4f}, {Zl[:,1].max():.4f}]")
    for K, amp in onset_scan([1.5, 1.6, K_star(), 1.75, 1.8, 2.0, 2.5]): print(f"K={K:.4f}: late prey amplitude = {amp:.5f}")
