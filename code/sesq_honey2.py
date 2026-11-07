r"""General sesquitruncated honeycomb from a Voronoi tiling (fcc -> rhombic dodecahedral,
hcp -> trapezo-rhombic dodecahedral). Vertices of order 4 (tetrahedral figure) are truncated,
vertices of order 8 (cubical figure) are sesquitruncated.

    python3 sesq_honey2.py hcp   -> fit + figs_hd/sesq_trd_honeycomb.pdf
"""
import sys, itertools, numpy as np
from collections import defaultdict
from scipy.spatial import Voronoi, ConvexHull
from scipy.optimize import minimize
from sesq_honey import hull_faces


def lattice(kind, R=4):
    P = []
    if kind == "fcc":
        for i, j, k in itertools.product(range(-R, R + 1), repeat=3):
            if (i + j + k) % 2 == 0:
                P.append((i, j, k))
        return np.array(P, float)
    a1, a2 = np.array([1, 0, 0.]), np.array([0.5, np.sqrt(3) / 2, 0])
    c = np.sqrt(8 / 3)
    for i, j, k in itertools.product(range(-R, R + 1), repeat=3):
        off = (a1 + a2) / 3 if k % 2 else 0
        P.append(i * a1 + j * a2 + k * c / 2 * np.array([0, 0, 1.]) + off)
    return np.array(P)


def tiling(kind):
    X = lattice(kind)
    vor = Voronoi(X)
    d = np.linalg.norm(X, axis=1)
    cells = {}
    for p in np.argsort(d)[:220]:                       # cells near the origin
        reg = vor.regions[vor.point_region[p]]
        if -1 in reg or not reg:
            continue
        cells[p] = reg
    V = vor.vertices
    # combinatorics: faces of each cell, global edges, vertex degree
    keyv = {}
    def kv(i):
        k = tuple(np.round(V[i], 6))
        return keyv.setdefault(k, len(keyv))
    gv = {}
    cellfaces = {}
    for p, reg in cells.items():
        pts = V[reg]
        F = hull_faces(pts)
        cellfaces[p] = [[kv(reg[i]) for i in f] for f in F]
        for i in reg:
            gv[kv(i)] = V[i]
    G = np.array([gv[i] for i in range(len(keyv))])
    nb = defaultdict(set)
    for F in cellfaces.values():
        for f in F:
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                nb[a].add(b); nb[b].add(a)
    return X, G, cellfaces, nb


def sesq_cell(G, F, cub, t, s, t4):
    pts = []
    for f in F:
        n = len(f)
        for i, v in enumerate(f):
            a, b = G[f[i - 1]], G[f[(i + 1) % n]]
            tt = t4 if v in cub else t
            for w in (a, b):
                pts.append(G[v] + tt * (w - G[v]))
            if v in cub:
                pts.append(G[v] + s * ((a + b) / 2 - G[v]))
    P = np.unique(np.round(np.array(pts), 9), axis=0)
    return P, hull_faces(P)


def vertex_cells(G, v, nb, faces_at, cub, t, s, t4):
    if v not in cub:
        P = np.array([G[v] + t * (G[w] - G[v]) for w in nb[v]])
        return [(P, hull_faces(P))]
    flags = {}
    for f in faces_at[v]:
        i = f.index(v); a, b = f[i - 1], f[(i + 1) % len(f)]
        flags[frozenset((a, b))] = G[v] + s * ((G[a] + G[b]) / 2 - G[v])
    Q = np.array(list(flags.values()))
    out = [(Q, hull_faces(Q))]
    for w in nb[v]:
        T = np.array([G[v] + t4 * (G[w] - G[v])] + [p for k, p in flags.items() if w in k])
        out.append((T, hull_faces(T)))
    return out


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "hcp"
    X, G, CF, nb = tiling(kind)
    deg = {v: len(nb[v]) for v in nb}
    c0 = min(CF, key=lambda p: np.linalg.norm(X[p]))
    F0 = CF[c0]
    full = {v for v in deg if np.linalg.norm(G[v]) < 1.2}       # vertices with complete stars
    cub = {v for v in deg if deg[v] == 8}
    print(kind, "central cell faces", sorted(len(f) for f in F0), "degrees", sorted(set(deg[v] for v in full)))
    faces_at = defaultdict(list)
    for F in CF.values():
        for f in F:
            for v in f:
                if f not in faces_at[v]:
                    faces_at[v].append(f)
    # dedupe faces shared by two cells
    for v in faces_at:
        seen, uniq = set(), []
        for f in faces_at[v]:
            k = frozenset(f)
            if k not in seen:
                seen.add(k); uniq.append(f)
        faces_at[v] = uniq
    vs = [v for v in set(i for f in F0 for i in f)]

    def ratio(x):
        t, s, t4 = x
        if not (0.02 < t < 0.48 and 0.02 < t4 < 0.9 and t + t4 < 0.97 and 0.02 < s < 0.98):
            return 9.0
        P, F = sesq_cell(G, F0, cub, t, s, t4)
        L = []
        for Q, H in [(P, F)] + [c for v in vs for c in vertex_cells(G, v, nb, faces_at, cub, t, s, t4)]:
            for f in H:
                for i in range(len(f)):
                    L.append(np.linalg.norm(Q[f[i]] - Q[f[(i + 1) % len(f)]]))
        L = np.array(L)
        return L.max() / L.min()

    best = None
    for x0 in itertools.product((0.2, 0.3), (0.3, 0.5), (0.3, 0.5)):
        o = minimize(ratio, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 3000})
        if best is None or o.fun < best.fun:
            best = o
    t, s, t4 = best.x
    P, F = sesq_cell(G, F0, cub, t, s, t4)
    from collections import Counter
    print("best ratio %.5f at t=%.4f s=%.4f t4=%.4f; sesq cell V %d faces %s"
          % (best.fun, t, s, t4, len(P), dict(Counter(len(f) for f in F))))
    np.savez("sesq_%s_params.npz" % kind, x=best.x)
    # render: cells around one cubical vertex of the central cell
    import render_hd as R
    v = min((u for u in vs if u in cub), key=lambda u: np.linalg.norm(G[u]))
    cells = [sesq_cell(G, CF[p], cub, t, s, t4) for p in CF if v in {i for f in CF[p] for i in f}]
    cells += vertex_cells(G, v, nb, faces_at, cub, t, s, t4)
    for w in nb[v]:
        cells += vertex_cells(G, w, nb, faces_at, cub, t, s, t4)
    small = vertex_cells(G, v, nb, faces_at, cub, t, s, t4)
    for w in nb[v]:
        small += vertex_cells(G, w, nb, faces_at, cub, t, s, t4)
    big = [sesq_cell(G, CF[p], cub, t, s, t4) for p in CF if v in {i for f in CF[p] for i in f}]
    R.cluster(big, small, G[v], "sesq_%s_honeycomb" % ("trd" if kind == "hcp" else "rd2"))
