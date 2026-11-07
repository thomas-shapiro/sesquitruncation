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
        if isinstance(rgb, str):
            style = "black!45,line width=0.2pt,dash pattern=on 0.8pt off 0.8pt" if rgb == "GHOST" else "red,line width=0.15pt"
            out.append("\\draw[" + style + "] (%.4f,%.4f) -- (%.4f,%.4f);"
                       % (P2[0, 0], P2[0, 1], P2[1, 0], P2[1, 1]))
            continue
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
        if lw == 0:
            out.append("\\fill[%s,line join=round] %s -- cycle;" % (cols[key], pts))
        else:
            out.append("\\filldraw[fill=%s,draw=black!80,line width=%.2fpt,line join=round] %s -- cycle;" % (cols[key], lw, pts))
    defs = ["\\definecolor{%s}{rgb}{%s}" % (v, k) for k, v in cols.items()]
    return defs + out


def add_cell_culled(items, Y, F, rgb, R):
    """convex cell: only its front faces, all at the depth of the cell centre (painter by cells)"""
    X = Y @ R.T
    vs = sorted({i for f in F for i in f})
    c = X[vs].mean(0)
    for f in F:
        Q = X[f]
        n = np.cross(Q[1] - Q[0], Q[2] - Q[0]); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ (Q.mean(0) - c) < 0:
            n = -n
        if n[2] <= 1e-9:
            continue
        items.append((Q[:, 2].mean(), Q[:, :2], rgb, n))


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
    rank = None
    if os.environ.get("CULL"):
        # exact back-to-front order of the cells of the Schlegel diagram (they tile a convex region):
        # for every 2-face shared by two inner cells, the cell on its far side is drawn first
        Z0 = Y @ R.T
        inner = [i for i, c in enumerate(C) if c is not cb]
        owner = {}
        for i in inner:
            for f in F:
                if set(f) <= C[i]:
                    owner.setdefault(frozenset(f), []).append(i)
        from collections import defaultdict, deque
        succ = defaultdict(set); indeg = {i: 0 for i in inner}
        for fk, own in owner.items():
            if len(own) != 2:
                continue
            f = sorted(fk); Q = Z0[f]
            nn = np.linalg.svd(Q - Q.mean(0))[2][2]
            if abs(nn[2]) < 1e-9:
                continue
            if nn[2] < 0: nn = -nn                       # towards the viewer
            i, j = own
            si = nn @ (Z0[sorted(C[i])].mean(0) - Q.mean(0))
            back, front = (i, j) if si < 0 else (j, i)
            if front not in succ[back]:
                succ[back].add(front); indeg[front] += 1
        dq = deque(i for i in inner if indeg[i] == 0); order = []
        while dq:
            i = dq.popleft(); order.append(i)
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0: dq.append(j)
        if len(order) != len(inner):
            print("WARNING: cycle in the visibility order", len(order), len(inner))
            order += [i for i in inner if i not in order]
        rank = {i: r for r, i in enumerate(order)}
    for ci, (c, kd) in enumerate(zip(C, kinds)):
        if kd in hide or (kd.startswith("tet") and "tet" in hide):
            continue
        sub = [f for f in F if set(f) <= c]
        if rank is not None:
            tmp = []
            add_cell_culled(tmp, Y, sub, COL[kd], R)
            items += [(rank[ci],) + it[1:] for it in tmp]
        else:
            add_cell(items, Y, sub, COL[kd], R)
    # all edges of the polytope, so that the hidden cells still connect the picture
    Z = Y @ R.T
    elw = float(os.environ.get("ELW", "0.3"))
    if os.environ.get("EXACTE"):
        # exact visibility: ray-cast sample points of every edge against the front faces of the shown
        # cells, and draw the visible runs on top of everything
        fr = []
        for c, kd in zip(C, kinds):
            if kd in hide or (kd.startswith("tet") and "tet" in hide):
                continue
            cen = Z[sorted(c)].mean(0)
            for f in F:
                if not set(f) <= c:
                    continue
                Q = Z[f]; nn = np.cross(Q[1] - Q[0], Q[2] - Q[0]); nn /= np.linalg.norm(nn)
                if nn @ (Q.mean(0) - cen) < 0: nn = -nn
                if nn[2] > float(os.environ.get("EDGEON", "1e-9")):
                    fr.append((Q, nn, Q[:, :2].min(0), Q[:, :2].max(0)))
        zr = Z[:, 2].max() - Z[:, 2].min()
        tol2d = 1e-4 * (Z[:, :2].max() - Z[:, :2].min())
        def hidden(p):
            for Q, nn, lo, hi in fr:
                if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]):
                    continue
                # strictly inside (distance > tol from every side): points on a face's
                # boundary are never hidden by that face
                s = 0; ok = True
                for i in range(len(Q)):
                    u, w = Q[i, :2], Q[(i + 1) % len(Q), :2]
                    L = np.hypot(*(w - u))
                    if L < 1e-12: continue
                    d = ((w[0] - u[0]) * (p[1] - u[1]) - (w[1] - u[1]) * (p[0] - u[0])) / L
                    if abs(d) <= tol2d: ok = False; break
                    s |= 1 if d > 0 else 2
                if not ok or s == 3: continue
                zf = Q[0, 2] - (nn[0] * (p[0] - Q[0, 0]) + nn[1] * (p[1] - Q[0, 1])) / nn[2]
                if zf > p[2] + 2e-3 * zr:
                    return True
            return False
        top = 1e9
        ns = 60
        for a, b in E:
            pts = [Z[a] + (Z[b] - Z[a]) * j / ns for j in range(ns + 1)]
            vis = [not hidden(p) for p in pts]
            j = 0
            ghost = os.environ.get("GHOST")
            while j <= ns:
                k = j
                while k + 1 <= ns and vis[k + 1] == vis[j]: k += 1
                lo, hi = max(j - 1, 0) if not vis[j] else j, min(k + 1, ns) if not vis[j] else k
                if vis[j] and k > j:
                    items.append((top, np.array([pts[j][:2], pts[k][:2]]), None, elw))
                elif not vis[j] and ghost:
                    items.append((top - 1, np.array([pts[lo][:2], pts[hi][:2]]), "GHOST", 0))
                j = k + 1
        if os.environ.get("DBGALL"):
            for a, b in E:
                items.append((2e9, Z[[a, b], :2], "RED", 0))
        Xs = np.vstack([it[1] for it in items])
        finish(tikz_faces(items, lw=0.0), Xs, name, width_cm=9)
        return
    nseg = int(os.environ.get("ESEG", "1"))
    eps = float(os.environ.get("EEPS", "0")) * (Z[:, 2].max() - Z[:, 2].min())
    for a, b in E:
        for j in range(nseg):
            p, q = Z[a] + (Z[b] - Z[a]) * j / nseg, Z[a] + (Z[b] - Z[a]) * (j + 1) / nseg
            items.append(((p[2] + q[2]) / 2 + 1e-6 + eps, np.array([p[:2], q[:2]]), None, elw))
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
