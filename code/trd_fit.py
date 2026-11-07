"""sesquitruncated trapezo-rhombic dodecahedral honeycomb with one parameter per symmetry class:
dart fraction per (vertex order, edge length), flag depth per (face shape, the two edge lengths at the corner).
Minimise the edge ratio over the central cell + vertex cells of its vertices."""
import itertools, numpy as np
from collections import defaultdict, Counter
from scipy.optimize import minimize
from sesq_honey import hull_faces
from sesq_honey2 import tiling

X, G, CF, nb = tiling("hcp")
cub = {v for v in nb if len(nb[v]) == 8}
seen, faces_at = set(), defaultdict(list)
for F in CF.values():
    for f in F:
        k = frozenset(f)
        if k in seen:
            continue
        seen.add(k)
        for v in f:
            faces_at[v].append(f)
lenkey = lambda a, b: round(float(np.linalg.norm(G[a] - G[b])), 4)
fshape = lambda f: tuple(sorted(lenkey(f[i], f[(i + 1) % len(f)]) for i in range(len(f))))
dkeys, fkeys = {}, {}
def dkey(v, w):
    return dkeys.setdefault((len(nb[v]), lenkey(v, w)), len(dkeys))
def fkey(v, f):
    i = f.index(v)
    return fkeys.setdefault((fshape(f), tuple(sorted((lenkey(v, f[i - 1]), lenkey(v, f[(i + 1) % len(f)]))))), len(fkeys))
c0 = min(CF, key=lambda p: np.linalg.norm(X[p]))
cells_near = [p for p in CF if np.linalg.norm(X[p]) < 1.2]
vs = {v for p in cells_near for f in CF[p] for v in f if len(faces_at[v]) >= (6 if len(nb[v]) == 4 else 12)}
for v in vs | {v for p in cells_near for f in CF[p] for v in f}:
    for w in nb[v]:
        dkey(v, w)
    if v in cub:
        for f in faces_at[v]:
            fkey(v, f)
nd, nf = len(dkeys), len(fkeys)
print("dart classes", dkeys, "\nflag classes", fkeys)


def pts(x):
    d, s = x[:nd], x[nd:]
    dart = lambda v, w: G[v] + d[dkey(v, w)] * (G[w] - G[v])
    def flag(v, f):
        i = f.index(v); a, b = f[i - 1], f[(i + 1) % len(f)]
        return G[v] + s[fkey(v, f)] * ((G[a] + G[b]) / 2 - G[v])
    return dart, flag


def cells(x):
    dart, flag = pts(x)
    out = []
    for p in cells_near:
        P = []
        for f in CF[p]:
            for i, v in enumerate(f):
                P += [dart(v, f[i - 1]), dart(v, f[(i + 1) % len(f)])]
                if v in cub:
                    P.append(flag(v, f))
        P = np.unique(np.round(np.array(P), 9), axis=0)
        out.append((P, hull_faces(P), "big"))
    for v in vs:
        if v not in cub:
            P = np.array([dart(v, w) for w in nb[v]]); out.append((P, hull_faces(P), "tetS"))
            continue
        fl = {frozenset((f[f.index(v) - 1], f[(f.index(v) + 1) % len(f)])): flag(v, f) for f in faces_at[v]}
        Q = np.array(list(fl.values())); out.append((Q, hull_faces(Q), "cuboct"))
        for w in nb[v]:
            T = np.array([dart(v, w)] + [p for k, p in fl.items() if w in k]); out.append((T, hull_faces(T), "tetC"))
    return out


def ratio(x):
    if np.any(x <= 0.02) or np.any(x[:nd] >= 0.7) or np.any(x[nd:] >= 1.2):
        return 9.0
    L = []
    try:
        for P, F, kd in cells(x):
            if kd == "big" and len(P) != 72:
                return 5.0
            for f in F:
                for i in range(len(f)):
                    L.append(np.linalg.norm(P[f[i]] - P[f[(i + 1) % len(f)]]))
    except Exception:
        return 9.0
    L = np.array(L)
    return L.max() / L.min()


if __name__ == "__main__":
    t, s, t4 = np.load("sesq_hcp_params.npz")["x"]
    x0 = np.array([t if k[0] == 4 else t4 for k in dkeys] + [s] * nf)
    best = (ratio(x0), x0)
    print("start", best[0])
    rng = np.random.default_rng(0)
    for k in range(12):
        xs = best[1] * (1 + rng.normal(0, 0.08 if k else 0, len(x0)))
        o = minimize(ratio, xs, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-11, "maxiter": 6000, "maxfev": 6000})
        if o.fun < best[0]:
            best = (o.fun, o.x); print(k, "ratio %.6f" % o.fun, flush=True)
    np.save("trd_params.npy", best[1])
    print("final %.6f" % best[0], np.round(best[1], 5))
