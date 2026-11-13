"""M4: Hopf bifurcation analysis for the RM model."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from params import P, x_star, y_star, K_star
from jacobian import J_rm, J_num, tr, det, disc

def coex_J(Kv, p=P):
    return J_num(J_rm, x_star(p), y_star(p, Kv), p, Kv)

def trace_formula(Kv, p=P):
    xs, ys = x_star(p), y_star(p, Kv); a, h, r = p['a'], p['h'], p['r']
    return xs*(a*a*h*ys/(1 + a*h*xs)**2 - r/Kv)

def sweep(Ks, p=P):
    out = []
    for Kv in Ks:
        J = coex_J(Kv, p); ev = np.linalg.eigvals(J)
        out.append((Kv, tr(J), det(J), disc(J), ev))
    return out

def find_K_hopf(p=P):
    return brentq(lambda Kv: tr(coex_J(Kv, p)), x_star(p)*1.001, 20.0, xtol=1e-12)

def find_K_disc(p=P):
    return brentq(lambda Kv: disc(coex_J(Kv, p)), x_star(p)*1.001, K_star(p), xtol=1e-12)

def make_figs(outdir="../outputs"):
    Ks = np.linspace(x_star()*1.02, 4.0, 600); S = sweep(Ks)
    T = np.array([s[1] for s in S]); D = np.array([s[2] for s in S])
    ev = np.array([s[4] for s in S]); Ks_ = K_star()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(Ks, ev.real[:, 0], label="Re $\\lambda$"); ax[0].plot(Ks, np.abs(ev.imag[:, 0]), "--", label="|Im $\\lambda$|")
    ax[0].axvline(Ks_, color="k", ls=":", label=f"K*={Ks_:.4f}"); ax[0].axhline(0, c="gray", lw=.5)
    ax[0].set_xlabel("K"); ax[0].set_title("Coexistence eigenvalues vs K"); ax[0].legend()
    ax[1].plot(Ks, T, label="tr J"); ax[1].plot(Ks, D, label="det J"); ax[1].axhline(0, c="gray", lw=.5)
    ax[1].axvline(Ks_, color="k", ls=":"); ax[1].set_xlabel("K"); ax[1].set_title("Trace and determinant vs K"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/fig_5_eigenvalues_trace_vs_K.png", dpi=300); plt.close()

if __name__ == "__main__":
    Kh, Kd = find_K_hopf(), find_K_disc()
    print(f"K* analytic = {K_star():.10f}  brentq = {Kh:.10f}  |diff|={abs(Kh-K_star()):.2e}")
    print(f"discriminant changes sign at K = {Kd:.6f}")
    for Kv in [0.8, 1.0, 1.5, K_star(), 1.8, 2.5, 3.0]:
        J = coex_J(Kv); print(f"K={Kv:.4f} tr={tr(J):+.5f} det={det(J):.5f} disc={disc(J):+.5f} eig={np.round(np.linalg.eigvals(J),4)}")
    dT = np.diff([tr(coex_J(k)) for k in np.linspace(0.4, 5, 200)]); print("trace monotone increasing in K:", bool(np.all(dT > 0)))
    print("closed-form trace check @K=3:", trace_formula(3.0)); make_figs()
