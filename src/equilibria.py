"""M2: equilibria (SymPy + fsolve)."""
import numpy as np, sympy as sp
from scipy.optimize import fsolve
from params import P, rm, baseline, x_star, y_star

def symbolic():
    x, y, r, K, a, h, c, d = sp.symbols('x y r K a h c d', positive=True)
    f = a*x/(1 + a*h*x)
    xd = r*x*(1 - x/K) - f*y
    yd = c*f*y - d*y
    rm_sol = sp.solve([xd, yd], [x, y], dict=True)
    bx = r*x*(1 - x/K) - a*x*y; by = c*x*y - d*y
    base_sol = sp.solve([bx, by], [x, y], dict=True)
    return rm_sol, base_sol

def coexists(p, K):
    return p['c'] > p['d']*p['h'] and K > x_star(p)

def numeric(model, p, K, guesses=None):
    f = lambda z: model(0, z, p, K)
    guesses = guesses or [(0.1,0.1),(K,0.1),(0.5,0.5),(1,1),(2,1),(K/2,2),(3,3)]
    roots = []
    for g in guesses:
        s, info, ier, _ = fsolve(f, g, full_output=True)
        if ier == 1 and s[0] >= -1e-9 and s[1] >= -1e-9 and np.linalg.norm(f(s)) < 1e-10:
            s = np.maximum(s, 0)
            if not any(np.allclose(s, q, atol=1e-6) for q in roots):
                roots.append(s)
    return sorted(roots, key=lambda q: (q[0], q[1]))

if __name__ == "__main__":
    rs, bs = symbolic()
    print("RM symbolic:"); [print("  ", s) for s in rs]
    print("Baseline symbolic:"); [print("  ", s) for s in bs]
    print("\nK  | analytic (x*,y*) | fsolve roots")
    for name, K in P['K_test'].items():
        print(f"{name:5s} K={K}: analytic=({x_star(P):.6f},{y_star(P,K):.6f}) coexists={coexists(P,K)}")
        for q in numeric(rm, P, K): print("      fsolve:", np.round(q, 6))
