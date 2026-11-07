r"""Sesquitruncation of 4-polytopes whose vertex figures are tetrahedra or cubes, and of
honeycombs with tetrahedral / cubical vertex figures.

At a simplicial vertex v: darts (v,e) = v + t (w - v).
At a cubical vertex v:    darts as above, and flag corners (v,phi) for the 2-faces phi at v,
                          (v,phi) = v + s (m_phi - v), m_phi = mean of the two neighbours of v on phi.
4-polytope: sigma(P) = convex hull of all darts and flag corners (checked: every point is a vertex).

    python3 sesq4d.py   -> prints counts / cell census / edge ratio for the examples
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull
from collections import Counter, defaultdict


# ---------------------------------------------------------------- polytopes
def cell24():
    V = set()
    for i, j in itertools.combinations(range(4), 2):
        for a, b in itertools.product((1, -1), repeat=2):
            x = [0] * 4; x[i] = a; x[j] = b; V.add(tuple(x))
    return np.array(sorted(V), float)


def joined_pentachoron():
    """dual of the rectified 5-cell: cell centres of the rectified 5-cell, in the hyperplane sum=0 of R^5"""
    # rectified 5-cell vertices: permutations of (1,1,0,0,0); its cells: 5 tetrahedra (x_i = 0 ... ) and 5 octahedra
    # facet normals of the rectified 5-cell: x_i <= 1 (octahedra) and x_i >= 0 (tetrahedra)
    # dual vertices = facet normals scaled by 1/offset, after centring
    c = np.full(5, 0.4)
    P = []
    for i in range(5):                       # x_i <= 1  ->  (e_i - c) . (x - c) <= 1 - 0.4
        n = np.eye(5)[i] - 0.2
        P.append(n / 0.6)
    for i in range(5):                       # x_i >= 0  ->  -(e_i - .2) . (x - c) <= 0.4
        n = -(np.eye(5)[i] - 0.2)
        P.append(n / 0.4)
    P = np.array(P)
    B = np.linalg.svd(np.ones((1, 5)))[2][1:]   # orthonormal basis of sum=0
    return P @ B.T


def faces4(V, tol=1e-7):
    """edges, 2-faces (vertex cycles) and cells (vertex sets) of a 4-polytope from its hull"""
    H = ConvexHull(V)
    eq = {}
    for e in H.equations:
        key = tuple(np.round(e / np.linalg.norm(e[:4]), 6))
        eq[key] = e / np.linalg.norm(e[:4])
    planes = np.array(list(eq.values()))
    on = np.abs(V @ planes[:, :4].T + planes[:, 4]) < tol        # vertex x facet
    cells = [frozenset(np.where(on[:, k])[0]) for k in range(len(planes))]
    ridges = set()
    for a, b in itertools.combinations(range(len(cells)), 2):
        r = cells[a] & cells[b]
        if len(r) >= 3:
            X = V[list(r)]
            if np.linalg.matrix_rank(X[1:] - X[0], tol=1e-6) == 2:
                ridges.add(frozenset(r))
    edges = set()
    for a, b in itertools.combinations(range(len(V)), 2):
        common = on[a] & on[b]
        if common.sum() >= 3:
            Pm = planes[common][:, :4]
            if np.linalg.matrix_rank(Pm, tol=1e-6) == 3:
                edges.add((a, b))
    return sorted(edges), [order_cycle(V, f) for f in ridges], cells


def order_cycle(V, f):
    f = list(f)
    X = V[f]; c = X.mean(0)
    U = np.linalg.svd(X - c)[2][:2]
    ang = np.arctan2((X - c) @ U[1], (X - c) @ U[0])
    return [f[k] for k in np.argsort(ang)]


# ---------------------------------------------------------------- the operation
ABS = None   # if set: darts at absolute distance t*ABS from the vertex


def sesq_points(V, edges, faces, t, s, cubical, t4=None):
    """darts at fraction t (t4 at cubical vertices) along the edges; the flag corners of a
    cubical vertex v on one hyperplane orthogonal to v (centre at 0), at depth s * the
    shallowest neighbour-midpoint, so that they span the rectified vertex figure."""
    t4 = t if t4 is None else t4
    pts, lab = [], []
    for a, b in edges:
        for v, w in ((a, b), (b, a)):
            tt = t4 if v in cubical else t
            if ABS:
                d = V[w] - V[v]; tt = tt * ABS / np.linalg.norm(d)
            pts.append(V[v] + tt * (V[w] - V[v])); lab.append(("dart", v, (min(a, b), max(a, b))))
    mids = defaultdict(list)
    for k, f in enumerate(faces):
        n = len(f)
        for i, v in enumerate(f):
            if v in cubical:
                mids[v].append((k, (V[f[i - 1]] + V[f[(i + 1) % n]]) / 2))
    for v, ms in mids.items():
        u = V[v] / np.linalg.norm(V[v])
        h = s * min((V[v] - m) @ u for k, m in ms)
        for k, m in ms:
            lam = h / ((V[v] - m) @ u)
            pts.append(V[v] + lam * (m - V[v])); lab.append(("flag", v, k))
    return np.array(pts), lab


def census4(P, tol=1e-7):
    E, F, C = faces4(P, tol)
    cells = Counter()
    for c in C:
        sub = [f for f in F if set(f) <= c]
        cells[tuple(sorted(Counter(len(f) for f in sub).items()))] += 1
    L = np.array([np.linalg.norm(P[a] - P[b]) for a, b in E])
    return E, F, C, cells, L.max() / L.min()


def expected_cells(V, cubical, faces, C):
    return len(C) + len(V) + len(cubical) * 8   # sigma(cells) + vertex cells + cuboct + 8 tetra (cubical: +7 extra)


def best_ts(V, edges, faces, cubical, cells, grid=9):
    from scipy.optimize import minimize
    want = len(cells) + (len(V) - len(cubical)) + 9 * len(cubical)

    def f(x):
        t, s, t4 = x
        if not (0.02 < t < 0.48 and 0.02 < t4 < 0.48 and 0.02 < s < 1.5):
            return 99.0
        P, lab = sesq_points(V, edges, faces, t, s, cubical, t4)
        try:
            H = ConvexHull(P)
        except Exception:
            return 99.0
        if len(H.vertices) != len(P):
            return 50.0
        E2, F2, C2 = faces4(P)
        if len(C2) != want:
            return 20.0 + abs(len(C2) - want)
        L = np.array([np.linalg.norm(P[a] - P[b]) for a, b in E2])
        return L.max() / L.min()

    best = None
    for t in np.linspace(0.1, 0.45, grid):
        for s in np.linspace(0.2, 1.2, grid):
            r = f((t, s, t))
            if best is None or r < best[0]:
                best = (r, (t, s, t))
    o = minimize(f, best[1], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300})
    return (o.fun, *o.x) if o.fun < best[0] else (best[0], *best[1])


def quick_ratio(P):
    E, F, C = faces4(P)
    L = np.array([np.linalg.norm(P[a] - P[b]) for a, b in E])
    return L.max() / L.min()


def show(name, V):
    V = V - V.mean(0)
    E, F, C = faces4(V)
    nb = Counter(); [nb.update((a, b)) for a, b in E]
    cubical = {v for v in range(len(V)) if nb[v] == 8}
    print(name, "V", len(V), "E", len(E), "F2", len(F), "C", len(C),
          "vertex degrees", dict(Counter(nb.values())))
    return V, E, F, C, cubical


EXAMPLES = (("24-cell", cell24), ("joined pentachoron", joined_pentachoron))


def build(k, grid=7):
    name, fn = EXAMPLES[k]
    V, E, F, C, cub = show(name, fn())
    r, t, s, t4 = best_ts(V, E, F, cub, C, grid=grid)
    P, lab = sesq_points(V, E, F, t, s, cub, t4)
    return name, P, lab, (t, s, t4)


def build_jp(maxiter=60):
    """joined pentachoron with the simplicial vertices moved radially by a factor q; differential evolution"""
    from scipy.optimize import differential_evolution
    V0, E, F, C, cub = show("joined pentachoron", joined_pentachoron())
    simp = [v for v in range(len(V0)) if v not in cub]
    want = len(C) + len(simp) + 9 * len(cub)

    def mk(x):
        q, t, s, t4 = x
        V = V0.copy(); V[simp] *= q
        return sesq_points(V, E, F, t, s, cub, t4)

    def f(x):
        P, lab = mk(x)
        try:
            H = ConvexHull(P)
        except Exception:
            return 99.0
        if len(H.vertices) != len(P):
            return 50.0
        E2, F2, C2 = faces4(P)
        if len(C2) != want:
            return 20.0 + abs(len(C2) - want)
        L = np.array([np.linalg.norm(P[a] - P[b]) for a, b in E2])
        return L.max() / L.min()

    from scipy.optimize import minimize
    rng = np.random.default_rng(1)
    o = None
    for k in range(maxiter):
        x0 = np.array([1.0, 0.224, 0.353, 0.441]) if k == 0 else \
            (o.x if o.fun < 10 else np.array([1.0, 0.224, 0.353, 0.441])) * (1 + rng.normal(0, 0.12, 4))
        r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 400})
        if o is None or r.fun < o.fun:
            o = r
            print("   restart %d ratio %.4f" % (k, r.fun), np.round(r.x, 4), flush=True)
    P, lab = mk(o.x)
    print("  jp best ratio %.4f at q,t,s,t4 = %s" % (o.fun, np.round(o.x, 4)), flush=True)
    return P, lab, o.x


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "jp":
    P, lab, x = build_jp()
    E2, F2, C2, cells, rr = census4(P)
    print("  V %d E %d F %d C %d ratio %.4f" % (len(P), len(E2), len(F2), len(C2), rr), dict(cells))
    np.save("sesq4d_1.npy", P)
elif __name__ == "__main__":
    import sys, json
    for k in range(len(EXAMPLES)):
        name, P, lab, (t, s, t4) = build(k, int(sys.argv[1]) if len(sys.argv) > 1 else 7)
        E2, F2, C2, cells, rr = census4(P)
        print("  sesq: t=%.3f s=%.3f t4=%.3f  V %d E %d F %d C %d  ratio %.4f" % (t, s, t4, len(P), len(E2), len(F2), len(C2), rr))
        np.save("sesq4d_%d.npy" % k, P)
        for k, v in sorted(cells.items(), key=lambda kv: -kv[1]):
            print("   ", v, "x", dict(k))
