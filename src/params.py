"""M1: shared parameters and model functions."""
import numpy as np

P = dict(r=1.0, a=1.0, h=1.0, c=2.0, d=0.5, K_test=None)

def x_star(p=P):
    return p['d'] / (p['a'] * (p['c'] - p['d'] * p['h']))

def K_star(p=P):
    return 2 * x_star(p) + 1 / (p['a'] * p['h'])

def y_star(p, K):
    xs = x_star(p)
    return p['r'] / p['a'] * (1 - xs / K) * (1 + p['a'] * p['h'] * xs)

# three test values: below, near, well above K*
P['K_test'] = dict(below=1.0, near=K_star(P), above=3.0)

def baseline(t, z, p, K):
    x, y = z
    return [p['r']*x*(1 - x/K) - p['a']*x*y, p['c']*x*y - p['d']*y]

def rm(t, z, p, K):
    x, y = z
    f = p['a']*x/(1 + p['a']*p['h']*x)
    return [p['r']*x*(1 - x/K) - f*y, p['c']*f*y - p['d']*y]

if __name__ == "__main__":
    print("x* =", x_star(), " K* =", K_star(), " tests:", P['K_test'])
    assert P['c'] > P['d']*P['h']
