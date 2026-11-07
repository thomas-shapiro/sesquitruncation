"""trd_fit.py plus one more parameter: the trapezo-rhombic tiling is scaled by lam along its c-axis
(an affine deformation, still a face-to-face convex tiling with the same combinatorics).
    python3 trd_fit7.py  -> trd7_params.npy, prints ratio and cell anisotropy"""
import os, sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trd_fit as T
G0 = T.G.copy()
# freeze the classes computed on the undeformed tiling
DK = {}
FK = {}
nd0, nf0 = len(T.dkeys), len(T.fkeys)
for p in T.cells_near:
    for f in T.CF[p]:
        for v in f:
            for w in T.nb[v]:
                DK[(v, w)] = T.dkey(v, w)
            if v in T.cub:
                FK[(v, frozenset(f))] = T.fkey(v, f)
for v in T.vs:
    for w in T.nb[v]:
        DK[(v, w)] = T.dkey(v, w)
    if v in T.cub:
        for f in T.faces_at[v]:
            FK[(v, frozenset(f))] = T.fkey(v, f)
assert (len(T.dkeys), len(T.fkeys)) == (nd0, nf0)
len0 = lambda a, b: round(float(np.linalg.norm(G0[a] - G0[b])), 4)
shape0 = lambda f: tuple(sorted(len0(f[i], f[(i + 1) % len(f)]) for i in range(len(f))))


def _dkey(v, w):
    k = DK.get((v, w))
    return k if k is not None else T.dkeys[(len(T.nb[v]), len0(v, w))]


def _fkey(v, f):
    k = FK.get((v, frozenset(f)))
    if k is None:
        i = f.index(v)
        k = T.fkeys[(shape0(f), tuple(sorted((len0(v, f[i - 1]), len0(v, f[(i + 1) % len(f)])))))]
    return k


T.dkey, T.fkey = _dkey, _fkey


def setlam(lam):
    T.G[:] = G0
    T.G[:, 2] *= lam


def ratio7(y):
    if not 0.5 < y[-1] < 1.5:
        return 9.0
    setlam(y[-1])
    return T.ratio(y[:-1])


def aniso(y):
    setlam(y[-1])
    acc = {}
    for P, F, kd in T.cells(y[:-1]):
        Q = P - P.mean(0); C = Q.T @ Q; C /= np.trace(C)
        acc.setdefault(kd, []).append(C)
    return {k: round(float(np.mean([c[2, 2] / c[0, 0] for c in v])), 3) for k, v in acc.items()}


if __name__ == "__main__":
    x6 = np.load("trd_params.npy")
    best = (ratio7(np.r_[x6, 1.0]), np.r_[x6, 1.0])
    print("start", best[0], aniso(best[1]), flush=True)
    rng = np.random.default_rng(1)
    for k in range(16):
        ys = best[1] * (1 + rng.normal(0, 0.06 if k else 0, len(best[1])))
        if k % 4 == 1:
            ys[-1] = rng.uniform(0.7, 1.0)
        o = minimize(ratio7, ys, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 8000, "maxfev": 8000})
        if o.fun < best[0]:
            best = (o.fun, o.x); print(k, "ratio %.6f lam %.4f" % (o.fun, o.x[-1]), flush=True)
    np.save("trd7_params.npy", best[1])
    print("final %.6f" % best[0], np.round(best[1], 5), aniso(best[1]))
