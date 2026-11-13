"""M5: from-scratch RK4 integrator, validation and convergence."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from params import P, rm, x_star, y_star

def rk4_step(f, t, z, h, p, K):
    k1 = np.array(f(t, z, p, K))
    k2 = np.array(f(t + h/2, z + h/2*k1, p, K))
    k3 = np.array(f(t + h/2, z + h/2*k2, p, K))
    k4 = np.array(f(t + h, z + h*k3, p, K))
    return z + h/6*(k1 + 2*k2 + 2*k3 + k4)

def rk4_solve(f, z0, t0, t1, h, p, K):
    n = int(round((t1 - t0)/h)); t = t0 + h*np.arange(n + 1)
    Z = np.empty((n + 1, len(z0))); Z[0] = z0
    for i in range(n): Z[i+1] = rk4_step(f, t[i], Z[i], h, p, K)
    return t, Z

def reference(f, z0, t1, p, K, teval):
    s = solve_ivp(lambda t, z: f(t, z, p, K), (0, t1), z0, method="DOP853", rtol=1e-13, atol=1e-14, t_eval=teval)
    return s.y.T

def convergence(z0=(0.5, 0.5), K=3.0, T=10.0, hs=(0.2, 0.1, 0.05, 0.025, 0.0125)):
    ref = reference(rm, z0, T, P, K, [T])[-1]; errs = []
    for h in hs:
        _, Z = rk4_solve(rm, np.array(z0), 0, T, h, P, K); errs.append(np.linalg.norm(Z[-1] - ref))
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    return np.array(hs), np.array(errs), slope

def make_figs(outdir="../outputs"):
    hs, errs, slope = convergence()
    plt.figure(figsize=(5, 4)); plt.loglog(hs, errs, "o-", label=f"RK4 (slope {slope:.2f})")
    plt.loglog(hs, errs[0]*(hs/hs[0])**4, "k--", label="slope 4 reference")
    plt.xlabel("h"); plt.ylabel("error at t=10"); plt.legend(); plt.grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig(f"{outdir}/fig_6_rk4_convergence.png", dpi=300); plt.close()
    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
    for a_, (nm, K) in zip(ax, P['K_test'].items()):
        T = 150; t, Z = rk4_solve(rm, np.array([0.5, 0.5]), 0, T, 0.01, P, K)
        a_.plot(t, Z[:, 0], label="prey x"); a_.plot(t, Z[:, 1], label="predator y")
        a_.set_title(f"K = {K:.4f} ({nm})"); a_.set_xlabel("t"); a_.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(f"{outdir}/fig_6_time_series.png", dpi=300); plt.close()

if __name__ == "__main__":
    hs, errs, slope = convergence()
    for h, e in zip(hs, errs): print(f"h={h:<7} error={e:.3e}")
    print(f"log-log slope = {slope:.3f}")
    for nm, K in P['K_test'].items():
        t, Z = rk4_solve(rm, np.array([0.5, 0.5]), 0, 100, 0.01, P, K)
        ref = reference(rm, [0.5, 0.5], 100, P, K, t)
        print(f"K={K:.4f} max|RK4-DOP853|={np.abs(Z-ref).max():.2e}  min x={Z[:,0].min():.4f} min y={Z[:,1].min():.4f}")
    make_figs()
