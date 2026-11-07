"""Sesquitruncation as an exact combinatorial map operation.

Parent P: any polyhedral map whose faces are only triangles and squares.
At a parent vertex v with t triangle-corners and s square-corners (degree d=t+s)
the child has a big face of size

        n_v = d + t + 2s = 2t + 3s

Child pieces
  X(e, f)  : one vertex per (parent edge, adjacent face) pair   -> 2E_P vertices, type 3.n.n
  Y(f, v)  : one vertex per (parent SQUARE f, corner v)         -> 4S_P vertices, type 3.4.3.n
  big face at v, triangle per parent triangle, square per parent square,
  plus 4 triangles per parent square (one per square edge).

  V_child = 2E_P + 4S_P        F_child = V_P + T_P + 5 S_P
"""
import sys, os
import numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, os.path.expanduser('~/nmj'))
from mapcheck import orient, validate


def rotation(F, nv):
    """at each vertex: cyclic list of neighbours, and the face sitting between
    consecutive neighbours.  F must already be consistently oriented."""
    nxt = defaultdict(dict)        # v -> {prev_neighbour: next_neighbour}
    cf = defaultdict(dict)         # v -> {prev_neighbour: face index}
    for fi, f in enumerate(F):
        k = len(f)
        for i in range(k):
            p, v, n = f[(i - 1) % k], f[i], f[(i + 1) % k]
            nxt[v][p] = n
            cf[v][p] = fi
    rot = {}
    for v in range(nv):
        a0 = next(iter(nxt[v]))
        a = a0
        nbrs, faces = [], []
        while True:
            nbrs.append(a)
            faces.append(cf[v][a])
            a = nxt[v][a]
            if a == a0:
                break
        rot[v] = (nbrs, faces)     # faces[i] sits between edge to nbrs[i] and edge to nbrs[i+1]
    return rot


def sesq(F, nv):
    """returns (child_faces, child_nv, info) or (None, None, reason)"""
    F = orient(F)
    ks = set(len(f) for f in F)
    if not ks <= {3, 4}:
        return None, None, f'parent has faces {sorted(ks)}, need only 3,4'
    rot = rotation(F, nv)
    S = [fi for fi, f in enumerate(F) if len(f) == 4]
    T = [fi for fi, f in enumerate(F) if len(f) == 3]

    ids = {}

    def vid(key):
        if key not in ids:
            ids[key] = len(ids)
        return ids[key]

    def X(u, w, fi):
        return vid(('X', frozenset((u, w)), fi))

    def Y(fi, v):
        return vid(('Y', fi, v))

    big = {}
    nsize = {}
    for v in range(nv):
        nbrs, faces = rot[v]
        d = len(nbrs)
        seq = []
        for i in range(d):
            a = nbrs[i]
            f_prev = faces[(i - 1) % d]      # face between edge i-1 and edge i
            f_next = faces[i]                # face between edge i and edge i+1
            seq.append(X(v, a, f_prev))
            seq.append(X(v, a, f_next))
            if len(F[f_next]) == 4:
                seq.append(Y(f_next, v))
        big[v] = seq
        nsize[v] = len(seq)

    child = [list(seq) for seq in big.values()]

    for fi in T:
        f = F[fi]
        child.append([X(f[i], f[(i + 1) % 3], fi) for i in range(3)])

    for fi in S:
        f = F[fi]
        child.append([Y(fi, v) for v in f])
        for i in range(4):
            u, w = f[i], f[(i + 1) % 4]
            child.append([X(u, w, fi), Y(fi, u), Y(fi, w)])

    cnv = len(ids)
    E_P = len({frozenset((f[i], f[(i + 1) % len(f)])) for f in F for i in range(len(f))})
    info = dict(V_P=nv, E_P=E_P, T_P=len(T), S_P=len(S),
                predicted_V=2 * E_P + 4 * len(S), predicted_F=nv + len(T) + 5 * len(S),
                bigsizes=dict(sorted(Counter(nsize.values()).items())))
    return child, cnv, info


# ------------------------------------------------------------------ parents
def hull_faces(pts, tol=1e-7):
    """convex hull with coplanar triangles merged into polygons; returns face list"""
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    h = ConvexHull(pts)
    groups = defaultdict(list)
    for eq, simp in zip(h.equations, h.simplices):
        key = tuple(np.round(eq / np.linalg.norm(eq[:3]), 6))
        groups[key].append(simp)
    faces = []
    for key, simps in groups.items():
        vs = sorted({int(i) for s in simps for i in s})
        n = np.array(key[:3])
        c = pts[vs].mean(0)
        # order the vertices around the face normal
        e1 = pts[vs[0]] - c; e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        ang = [np.arctan2(np.dot(pts[v] - c, e2), np.dot(pts[v] - c, e1)) for v in vs]
        order = [v for _, v in sorted(zip(ang, vs))]
        faces.append(order)
    return faces


def cuboctahedron():
    return hull_faces([p for p in
                       [(a, b, 0) for a in (1, -1) for b in (1, -1)] +
                       [(a, 0, c) for a in (1, -1) for c in (1, -1)] +
                       [(0, b, c) for b in (1, -1) for c in (1, -1)]])


def rhombicuboctahedron():
    s = 1 + np.sqrt(2)
    pts = set()
    for a in (1, -1):
        for b in (1, -1):
            for c in (s, -s):
                pts.update({(a, b, c), (a, c, b), (c, a, b)})
    return hull_faces(sorted(pts))


def snub_cube():
    # tribonacci constant
    t = 1.8392867552141612
    c1, c2, c3 = 1.0, 1 / t, t
    pts = []
    base = [(c1, c2, c3)]
    # even permutations with an even number of minus signs, plus odd perms with odd
    import itertools
    perms_even = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    perms_odd = [(0, 2, 1), (2, 1, 0), (1, 0, 2)]
    v = (c1, c2, c3)
    for signs in itertools.product((1, -1), repeat=3):
        nneg = sum(1 for s in signs if s < 0)
        for p in (perms_even if nneg % 2 == 0 else perms_odd):
            pts.append(tuple(signs[i] * v[p[i]] for i in range(3)))
    uniq = sorted(set(tuple(np.round(p, 9)) for p in pts))
    return hull_faces(uniq)


if __name__ == '__main__':
    from analyze import read_off
    from develop import canonical
    tests = [('cuboctahedron', cuboctahedron()),
             ('rhombicuboctahedron', rhombicuboctahedron()),
             ('snub cube', snub_cube())]
    for name, F in tests:
        nv = max(max(f) for f in F) + 1
        probs = validate(orient(F), nv)
        fc = Counter(len(f) for f in F)
        print(f"\n=== parent {name}: V={nv} F={len(F)} {dict(sorted(fc.items()))} problems={probs}")
        child, cnv, info = sesq(F, nv)
        if child is None:
            print('   ', info); continue
        print('   ', info)
        cf = Counter(len(f) for f in child)
        print(f"    child V={cnv} F={len(child)} faces={dict(sorted(cf.items()))}")
        out = canonical(child, cnv)
        if out is None:
            print('    solver failed'); continue
        V, d = out
        print(f"    RATIO {d['ratio']:.5f}  mid {d['midsphere_resid']:.1e} planar {d['planarity']:.1e} nonconvex {d['faces_with_vertices_outside']}")
