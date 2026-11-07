r"""Sesquitruncated rhombic dodecahedral honeycomb (cells = Voronoi cells of the fcc lattice).
Vertices of the honeycomb: half-integer points (order 4, tetrahedral figure) and integer points with
odd coordinate sum (order 8, cubical figure).  Darts at fraction t (t4 at cubical vertices) along the
edges, flag corners (v,phi) = v + s (m_phi - v), m_phi = mean of v's two neighbours on the rhombus phi.

    python3 sesq_honey.py   -> edge-ratio optimum and sesq_honey.npz (cells around one cubical vertex)
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

RDV = np.array([p for p in itertools.permutations((1, 0, 0))] + [p for p in itertools.permutations((-1, 0, 0))]
               + list(itertools.product((.5, -.5), repeat=3)), float)
RDV = np.unique(RDV, axis=0)


def hull_faces(X, tol=1e-7):
    H = ConvexHull(X)
    groups = {}
    for eq in H.equations:
        key = tuple(np.round(eq, 6))
        groups[key] = eq
    faces = []
    for eq in groups.values():
        idx = np.where(np.abs(X @ eq[:3] + eq[3]) < tol)[0]
        c = X[idx].mean(0)
        n = eq[:3]
        u = X[idx[0]] - c; u /= np.linalg.norm(u)
        w = np.cross(n, u)
        ang = np.arctan2((X[idx] - c) @ w, (X[idx] - c) @ u)
        faces.append([int(i) for i in idx[np.argsort(ang)]])
    return faces


def rd_cell(c):
    X = RDV + c
    return X, hull_faces(X)


def is_cubical(p):
    return np.allclose(p, np.round(p)) and int(round(sum(p))) % 2 == 1


def sesq_cell(c, t, s, t4):
    """sigma of the RD cell at centre c: returns points (n,3) and faces"""
    X, F = rd_cell(c)
    pts = []
    for f in F:
        n = len(f)
        for i, v in enumerate(f):
            a, b = X[f[i - 1]], X[f[(i + 1) % n]]
            for w in (a, b):
                tt = t4 if is_cubical(X[v]) else t
                pts.append(X[v] + tt * (w - X[v]))
            if is_cubical(X[v]):
                pts.append(X[v] + s * ((a + b) / 2 - X[v]))
    P = np.unique(np.round(np.array(pts), 9), axis=0)
    return P, hull_faces(P)


def vertex_cells(v, t, s, t4):
    """cells of sigma at a honeycomb vertex v: a tetrahedron (simplicial) or cuboctahedron + 8 tetrahedra"""
    if not is_cubical(v):
        nb = [v + d for d in itertools.product((.5, -.5), repeat=3)]
        nb = [w for w in nb if is_cubical(w)]
        P = np.array([v + t * (w - v) for w in nb])
        return [(P, hull_faces(P))]
    d = np.array([1, 0, 0]) if abs(v[0] - round(v[0])) < 1e-9 else None
    nb = [v + np.array(e) for e in itertools.product((.5, -.5), repeat=3)]
    darts = {tuple(np.round(w, 6)): v + t4 * (w - v) for w in nb}
    # rhombi at v: pairs of neighbours w, w' at distance 1 (they share a rhombus with v)
    flags = {}
    for w, w2 in itertools.combinations(nb, 2):
        if abs(np.linalg.norm(w - w2) - 1) < 1e-9:
            flags[(tuple(np.round(w, 6)), tuple(np.round(w2, 6)))] = v + s * ((w + w2) / 2 - v)
    Q = np.array(list(flags.values()))
    cells = [(Q, hull_faces(Q))]
    for w in nb:
        k = tuple(np.round(w, 6))
        fl = [p for (a, b), p in flags.items() if k in (a, b)]
        T = np.array([darts[k]] + fl)
        cells.append((T, hull_faces(T)))
    return cells


def edges_of(P, F):
    E = set()
    for f in F:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            E.add((min(a, b), max(a, b)))
    return [np.linalg.norm(P[a] - P[b]) for a, b in E]


def ratio(x):
    t, s, t4 = x
    if not (0.02 < t < 0.48 and 0.02 < t4 < 0.9 and t + t4 < 0.97 and 0.02 < s < 0.98):
        return 9.0
    P, F = sesq_cell(np.zeros(3), t, s, t4)
    if len(P) != 2 * 24 + 6 * 4 or len(F) != 12 + 8 + 6 + 24:
        return 5.0
    L = edges_of(P, F)
    for Q, G in vertex_cells(np.array([1., 0, 0]), t, s, t4) + vertex_cells(np.array([.5, .5, .5]), t, s, t4):
        L += edges_of(Q, G)
    L = np.array(L)
    return L.max() / L.min()


if __name__ == "__main__":
    best = None
    for x0 in itertools.product((0.2, 0.3, 0.4), (0.3, 0.5, 0.7), (0.2, 0.3, 0.4)):
        o = minimize(ratio, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 3000})
        if best is None or o.fun < best.fun:
            best = o
    t, s, t4 = best.x
    print("best ratio %.6f at t=%.5f s=%.5f t4=%.5f" % (best.fun, t, s, t4))
    P, F = sesq_cell(np.zeros(3), t, s, t4)
    from collections import Counter
    print("sesq RD cell: V", len(P), "faces", dict(Counter(len(f) for f in F)))
    np.savez("sesq_honey_params.npz", x=best.x)
