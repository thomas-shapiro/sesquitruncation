r"""Renders in the style of the 2023 paper (TikZ -> PDF):
  honeycombs: a large opaque patch, cells coloured by type, Lambert shading;
  4-polytopes: perspective projection with the largest cells hidden.
    python3 render_hd2.py rd | trd | 24 | jp [az el]
"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_hd import rot, finish, OUT

COL = {"big": (0.30, 0.42, 0.78), "cuboct": (0.85, 0.25, 0.22), "tetS": (0.95, 0.85, 0.20),
       "tetC": (0.95, 0.85, 0.20), "tetE": (0.95, 0.85, 0.20), "tet": (0.95, 0.85, 0.20)}


def view(az, el):
    """world z vertical on the page, seen from azimuth az and elevation el (degrees)"""
    return rot(0, el - 90) @ rot(az, 0)
L = np.array([-0.35, 0.55, 0.76]); L /= np.linalg.norm(L)


def tikz_faces(items, lw=0.25):
    """items: (depth, P2 (k,2), rgb, normal3) ; painter's order"""
    items.sort(key=lambda it: it[0])
    out = []
    cols = {}
    for z, P2, rgb, n in items:
        if rgb is None:
            out.append("\\draw[black!85,line width=%.2fpt,line cap=round] (%.4f,%.4f) -- (%.4f,%.4f);"
                       % (n, P2[0, 0], P2[0, 1], P2[1, 0], P2[1, 1]))
            continue
        sh = 0.55 + 0.45 * abs(n @ L)
        c = tuple(min(1, x * sh + 0.08) for x in rgb)
        key = "%.3f,%.3f,%.3f" % c
        if key not in cols:
            cols[key] = "c%d" % len(cols)
        pts = " -- ".join("(%.4f,%.4f)" % (x, y) for x, y in P2)
        out.append("\\filldraw[fill=%s,draw=black!80,line width=%.2fpt,line join=round] %s -- cycle;" % (cols[key], lw, pts))
    defs = ["\\definecolor{%s}{rgb}{%s}" % (v, k) for k, v in cols.items()]
    return defs + out


def add_cell(items, P, F, rgb, R):
    X = P @ R.T
    for f in F:
        Q = X[f]
        n = np.cross(Q[1] - Q[0], Q[2] - Q[0]); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        items.append((Q[:, 2].mean(), Q[:, :2], rgb, n))


def honey_rd(az, el, Rad=2.3):
    import sesq_honey as H
    t, s, t4 = np.load(os.path.join(os.path.dirname(__file__), "sesq_honey_params.npz"))["x"]
    R = view(az, el)
    items = []
    centres = [np.array(c, float) for c in itertools.product(range(-3, 4), repeat=3)
               if sum(c) % 2 == 0 and np.linalg.norm(c) <= Rad]
    for c in centres:
        P, F = H.sesq_cell(c, t, s, t4); add_cell(items, P, F, COL["big"], R)
    verts = {tuple(np.round(c + d, 6)) for c in centres for d in H.RDV}
    for v in verts:
        v = np.array(v)
        cells = H.vertex_cells(v, t, s, t4)
        for k, (P, F) in enumerate(cells):
            kind = "tetS" if not H.is_cubical(v) else ("cuboct" if k == 0 else "tetC")
            add_cell(items, P, F, COL[kind], R)
    X = np.vstack([it[1] for it in items])
    finish(tikz_faces(items), X, "sesq_rd_honeycomb", width_cm=10)


def honey_trd(az, el):
    import sesq_honey2 as H2
    X, G, CF, nb = H2.tiling("hcp")
    t, s, t4 = np.load(os.path.join(os.path.dirname(__file__), "sesq_hcp_params.npz"))["x"]
    cub = {v for v in nb if len(nb[v]) == 8}
    from collections import defaultdict
    faces_at = defaultdict(list)
    seen = set()
    for F in CF.values():
        for f in F:
            k = frozenset(f)
            if k in seen:
                continue
            seen.add(k)
            for v in f:
                faces_at[v].append(f)
    R = view(az, el)
    items = []
    keep = [p for p in CF if np.linalg.norm(X[p]) <= 2.2]
    for p in keep:
        P, F = H2.sesq_cell(G, CF[p], cub, t, s, t4); add_cell(items, P, F, COL["big"], R)
    vs = {v for p in keep for f in CF[p] for v in f}
    for v in vs:
        if len(nb[v]) not in (4, 8) or len(faces_at[v]) < (6 if len(nb[v]) == 4 else 12):
            continue
        cells = H2.vertex_cells(G, v, nb, faces_at, cub, t, s, t4)
        for k, (P, F) in enumerate(cells):
            kind = "tetS" if v not in cub else ("cuboct" if k == 0 else "tetC")
            if kind == "tetC":
                # polar (triangle of flag corners perpendicular to the c-axis) or equatorial
                n = np.cross(P[2] - P[1], P[3] - P[1]); n /= np.linalg.norm(n)
                kind = "tetC" if abs(n[2]) > 0.9 else "tetE"
            add_cell(items, P, F, COL[kind], R)
    Xs = np.vstack([it[1] for it in items])
    finish(tikz_faces(items), Xs, "sesq_trd_honeycomb", width_cm=10)


SCHLEGEL_CELL = os.environ.get("SCELL", "big")


def poly4(k, name, az, el, dist=1.15, hide="big"):
    import sesq4d as S
    P = np.load(os.path.join(os.path.dirname(__file__), "sesq4d_%d.npy" % k))
    P = P - P.mean(0)
    E, F, C = S.faces4(P)
    big_n = max(len(c) for c in C)
    kinds = []
    for c in C:
        sub = [f for f in F if set(f) <= c]
        kinds.append("big" if len(c) == big_n else ("cuboct" if len(sub) == 14 else "tet"))
    # perspective projection 4D -> 3D from an eye on the axis through the centre of a big cell
    from render_hd import schlegel
    cb = [c for c, kd in zip(C, kinds) if kd == SCHLEGEL_CELL][0]
    Y = schlegel(P, cb, dist)
    R = rot(az, el)
    if os.environ.get("VIEWSQ"):
        # look straight down a 4-fold axis of the outer cell: normal of one of its squares -> z
        sq = [f for f in F if set(f) <= cb and len(f) == int(os.environ.get("VIEWSQ"))][0]
        n = np.cross(Y[sq[1]] - Y[sq[0]], Y[sq[2]] - Y[sq[0]]); n /= np.linalg.norm(n)
        a = np.cross(n, [0, 0, 1.0]); sa = np.linalg.norm(a); ca = n[2]
        if sa > 1e-9:
            a /= sa
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = rot(az, el) @ (np.eye(3) + sa * K + (1 - ca) * K @ K)
    if os.environ.get("ROTY"):
        b = np.radians(float(os.environ["ROTY"]))
        R = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]) @ R
    if os.environ.get("DIAMOND"):
        # in-plane rotation that makes the (length-weighted) squares of the cuboctahedra diamonds
        Z = Y @ R.T; acc = 0j
        for c, kd in zip(C, kinds):
            if kd != "cuboct":
                continue
            for f in F:
                if set(f) <= c and len(f) == 4:
                    for i in range(4):
                        d = Z[f[(i + 1) % 4], :2] - Z[f[i], :2]
                        acc += np.hypot(*d) ** 2 * np.exp(4j * np.arctan2(d[1], d[0]))
        phi = np.angle(acc) / 4
        delta = (np.pi / 4 - phi + np.pi / 4) % (np.pi / 2) - np.pi / 4
        print("diamond rotation %.2f deg" % np.degrees(delta))
        R = rot(np.degrees(delta), 0) @ R
    # tetrahedra: those of simplicial vertices vs those around a cuboctahedron
    cub_v = set()
    for c, kd in zip(C, kinds):
        if kd == "cuboct":
            cub_v |= set(c)
    kinds = [("tetC" if len(set(c) & cub_v) == 3 else "tetS") if kd == "tet" else kd for c, kd in zip(C, kinds)]
    items = []
    for c, kd in zip(C, kinds):
        if kd in hide or (kd.startswith("tet") and "tet" in hide):
            continue
        sub = [f for f in F if set(f) <= c]
        add_cell(items, Y, sub, COL[kd], R)
    # all edges of the polytope, so that the hidden cells still connect the picture
    Z = Y @ R.T
    elw = float(os.environ.get("ELW", "0.3"))
    for a, b in E:
        items.append((Z[[a, b], 2].mean() + 1e-6, Z[[a, b], :2], None, elw))
    Xs = np.vstack([it[1] for it in items])
    finish(tikz_faces(items, lw=0.2), Xs, name, width_cm=9)


if __name__ == "__main__":
    w = sys.argv[1]
    az, el = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (24, 18)
    dist = float(sys.argv[4]) if len(sys.argv) > 4 else 1.15
    if w == "rd":
        honey_rd(az, el)
    elif w == "trd":
        honey_trd(az, el)
    elif w == "24":
        poly4(0, "sesq24cell_hidden", az, el, dist, hide=("big",))
    elif w == "jp":
        poly4(1, "sesqjp_hidden", az, el, dist, hide=tuple(os.environ.get("HIDE", "big,cuboct").split(",")))
