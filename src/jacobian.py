"""M3: Jacobians, trace/det/discriminant, classification."""
import numpy as np, sympy as sp
from params import P, x_star, y_star

x, y, r, K, a, h, c, d = sp.symbols('x y r K a h c d', positive=True)
f = a*x/(1 + a*h*x)
F_rm = sp.Matrix([r*x*(1-x/K) - f*y, c*f*y - d*y])
F_bl = sp.Matrix([r*x*(1-x/K) - a*x*y, c*x*y - d*y])
J_rm = F_rm.jacobian([x, y]); J_bl = F_bl.jacobian([x, y])

def J_num(Jsym, xv, yv, p, Kv):
    sub = {x: xv, y: yv, r: p['r'], a: p['a'], h: p['h'], c: p['c'], d: p['d'], K: Kv}
    return np.array(Jsym.subs(sub).evalf().tolist(), dtype=float)

def tr(J): return float(np.trace(J))
def det(J): return float(np.linalg.det(J))
def disc(J): return tr(J)**2 - 4*det(J)

def classify_point(J, tol=1e-9):
    T, D, Delta = tr(J), det(J), disc(J)
    if D < -tol: return "saddle"
    if abs(D) <= tol: return "non-hyperbolic (det = 0)"
    if abs(T) <= tol: return "centre-type (linear): Hopf candidate"
    if abs(Delta) <= tol: return ("degenerate stable node" if T < 0 else "degenerate unstable node")
    if Delta > 0: return "stable node" if T < 0 else "unstable node"
    return "stable spiral" if T < 0 else "unstable spiral"

if __name__ == "__main__":
    print("RM Jacobian:", sp.simplify(J_rm))
    print("Baseline Jacobian:", sp.simplify(J_bl))
    for nm, Kv in P['K_test'].items():
        xs, ys = x_star(P), y_star(P, Kv)
        print(f"\n--- RM, K={Kv} ({nm})")
        for lab, pt in [("(0,0)", (0,0)), ("(K,0)", (Kv,0)), ("(x*,y*)", (xs,ys))]:
            Jn = J_num(J_rm, *pt, P, Kv)
            print(f"{lab}: tr={tr(Jn):.5f} det={det(Jn):.5f} disc={disc(Jn):.5f} eig={np.round(np.linalg.eigvals(Jn),5)} -> {classify_point(Jn)}")
        # baseline interior
        xb, yb = P['d']/P['c'], P['r']/P['a']*(1 - P['d']/(P['c']*Kv))
        Jb = J_num(J_bl, xb, yb, P, Kv)
        print(f"Baseline interior ({xb:.3f},{yb:.3f}): tr={tr(Jb):.5f} det={det(Jb):.5f} -> {classify_point(Jb)}")
