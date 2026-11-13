"""Extra figures: equilibria schematic (M2) and trace-determinant plane (M3)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import P, x_star, y_star
from jacobian import J_rm, J_num, tr, det
from fields import prey_nullcline

def make(outdir="../outputs"):
    K = P['K_test']['above']; fig, ax = plt.subplots(figsize=(6, 4.6)); xx = np.linspace(0, K, 300)
    ax.plot(xx, prey_nullcline(xx, P, K), "b", label="prey nullcline"); ax.axvline(x_star(), c="r", label="predator nullcline")
    ax.axhline(0, c="g", lw=1, label="x-axis (y=0)"); ax.axvline(0, c="g", lw=1)
    for pt, lab in [((0, 0), "(0,0)"), ((K, 0), "(K,0)"), ((x_star(), y_star(P, K)), "(x*,y*)")]:
        ax.plot(*pt, "ko"); ax.annotate(lab, pt, textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(f"Equilibria, K={K}"); ax.legend(); plt.tight_layout()
    plt.savefig(f"{outdir}/fig_3_equilibria_schematic.png", dpi=300); plt.close()
    fig, ax = plt.subplots(figsize=(6, 4.6)); T = np.linspace(-2, 2, 400); ax.plot(T, T**2/4, "k", label=r"$tr^2=4\det$")
    ax.axhline(0, c="gray"); ax.axvline(0, c="gray")
    for nm, Kv in P['K_test'].items():
        for lab, pt, m in [("(0,0)", (0, 0), "s"), ("(K,0)", (Kv, 0), "^"), ("(x*,y*)", (x_star(), y_star(P, Kv)), "o")]:
            J = J_num(J_rm, *pt, P, Kv); ax.plot(tr(J), det(J), m, label=f"{lab}, K={Kv:.2f}")
    ax.set_xlabel("trace"); ax.set_ylabel("determinant"); ax.set_ylim(-1.5, 1); ax.legend(fontsize=6, ncol=2); ax.set_title("Trace-determinant plane")
    plt.tight_layout(); plt.savefig(f"{outdir}/fig_4_trace_determinant.png", dpi=300); plt.close()
