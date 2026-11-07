r"""Renders for the higher-dimensional section, as standalone TikZ -> PDF (grey shading, black edges).

    python3 render_hd.py 24      -> figs_hd/sesq24cell_schlegel.pdf, ..._ortho.pdf
    python3 render_hd.py jp      -> figs_hd/sesqjp_*.pdf
    python3 render_hd.py honey   -> figs_hd/sesq_rd_honeycomb.pdf
"""
import os, sys, itertools, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RENDER_OUT", os.path.join(HERE, "figs_hd"))
os.makedirs(OUT, exist_ok=True)
GREY = {3: 0.94, 4: 0.80}          # face brightness by size; big faces darker


def grey(n):
    return GREY.get(n, 0.60)


def rot(az, el):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def draw(X3, faces, alpha, light=True, lw=0.35, edge_depth=True):
    """X3: rotated 3D points (z toward the viewer). Painter's algorithm -> list of TikZ lines."""
    L = np.array([-0.4, 0.5, 0.75]); L /= np.linalg.norm(L)
    items = []
    for f in faces:
        P = X3[f]
        shade = 1.0
        if light:
            n = np.cross(P[1] - P[0], P[2] - P[0])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                n /= nn
                shade = 0.55 + 0.45 * abs(n @ L)
        items.append((P[:, 2].mean(), P[:, :2], grey(len(f)) * shade))
    items.sort(key=lambda it: it[0])
    zs = np.array([it[0] for it in items])
    z0, z1 = zs.min(), zs.max() + 1e-9
    out = []
    for z, P2, g in items:
        d = (z - z0) / (z1 - z0)
        ec = 0.35 + 0.65 * d if edge_depth else 1.0
        w = lw * (0.6 + 0.8 * d) if edge_depth else lw
        pts = " -- ".join("(%.4f,%.4f)" % (x, y) for x, y in P2)
        out.append("\\filldraw[fill=black!%d,fill opacity=%.3f,draw=black!%d,line width=%.3fpt,line join=round] %s -- cycle;"
                   % (round(100 * (1 - g)), alpha, round(100 * ec), w, pts))
    return out


def finish(lines, X2, name, width_cm=9.0):
    span = (X2.max(0) - X2.min(0)).max()
    sc = width_cm / span
    tex = ("\\documentclass[tikz,border=2pt]{standalone}\n\\begin{document}\n"
           "\\begin{tikzpicture}[x=%.4fcm,y=%.4fcm]\n%s\n\\end{tikzpicture}\n\\end{document}\n"
           % (sc, sc, "\n".join(lines)))
    base = os.path.join(OUT, name)
    open(base + ".tex", "w").write(tex)
    r = subprocess.run([os.path.expanduser("~/bin/tectonic"), name + ".tex"], capture_output=True, text=True, cwd=OUT)
    print("wrote", base + ".pdf" if r.returncode == 0 else "FAILED " + r.stderr[-600:], flush=True)


def schlegel(P, cell, depth=1.15):
    """perspective projection of the 4-polytope P (centred) into the hyperplane of `cell`"""
    c = sorted(cell)
    cen = P[c].mean(0)
    U = np.linalg.svd(P[c] - cen)[2]
    n = U[3] if U[3] @ cen > 0 else -U[3]
    d = n @ cen
    eye = n * d * depth
    Y = np.array([eye + (d - eye @ n) / ((x - eye) @ n) * (x - eye) for x in P]) - cen
    return Y @ U[:3].T


def draw4(X3, w, faces, alpha, lw=0.5):
    """like draw, but depth = 4D coordinate w (larger = nearer the viewer / outer); only the
    near half (w >= median) is filled, the far half is drawn as faint wire"""
    items = []
    for f in faces:
        items.append((w[f].mean(), X3[f][:, :2], grey(len(f)), len(f)))
    items.sort(key=lambda it: it[0])
    ws = np.array([it[0] for it in items]); w0, w1 = ws.min(), ws.max() + 1e-9
    out = []
    for z, P2, g, n in items:
        d = (z - w0) / (w1 - w0)
        a = alpha * (0.25 + 1.5 * d) if n > 4 else alpha * (0.15 + 0.6 * d)
        ec = 0.18 + 0.82 * d ** 1.5
        pts = " -- ".join("(%.4f,%.4f)" % (x, y) for x, y in P2)
        out.append("\\filldraw[fill=black!%d,fill opacity=%.3f,draw=black!%d,line width=%.3fpt,line join=round] %s -- cycle;"
                   % (round(100 * (1 - g)), min(a, 0.9), round(100 * ec), lw * (0.35 + 0.9 * d), pts))
    return out


def polychoron(tag, k, az=30, el=20, depth=1.15, seed=3):
    sys.path.insert(0, HERE)
    import sesq4d
    P = np.load(os.path.join(HERE, "sesq4d_%d.npy" % k))
    P = P - P.mean(0)
    E, F, C = sesq4d.faces4(P)
    big = max(C, key=len)
    Y = schlegel(P, big, depth)
    cen = P[sorted(big)].mean(0); u = cen / np.linalg.norm(cen)
    X3 = Y @ rot(az, el).T
    finish(draw4(X3, P @ u, F, alpha=0.22), X3[:, :2], tag + "_schlegel")
    # cell-first orthographic projection: drop the coordinate along the big cell's centre,
    # and use the big cell's own axes (from the Schlegel frame) so its symmetry shows
    c = sorted(big)
    B = np.linalg.svd(P[c] - P[c].mean(0))[2][:3].T
    X3 = (P @ B) @ rot(az, el).T
    finish(draw4(X3, P @ u, F, alpha=0.20), X3[:, :2], tag + "_ortho")


def cluster(big, small, centre, name, az=28, el=22, explode=0.0):
    """big: polyhedral cells (exploded away from `centre`, the front ones removed so the
    core is visible); small: vertex cells (not moved)."""
    R = rot(az, el)
    Xa, Fa, off = [], [], 0
    for P, F in big:
        d = P.mean(0) - centre
        if (d @ R.T)[2] > 0.25 * np.linalg.norm(d):      # toward the viewer: leave out
            continue
        Xa.append(P + explode * d); Fa += [[i + off for i in f] for f in F]; off += len(P)
    for P, F in small:
        Xa.append(P); Fa += [[i + off for i in f] for f in F]; off += len(P)
    X = np.vstack(Xa) @ R.T
    finish(draw(X, Fa, alpha=1.0, light=True, lw=0.4, edge_depth=False), X[:, :2], name)


def honeycomb_v2(az=28, el=22):
    sys.path.insert(0, HERE)
    import sesq_honey as H
    t, s, t4 = np.load(os.path.join(HERE, "sesq_honey_params.npz"))["x"]
    v = np.array([1.0, 0, 0])
    big = [H.sesq_cell(np.array(c, float), t, s, t4)
           for c in ([0, 0, 0], [2, 0, 0], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1])]
    small = H.vertex_cells(v, t, s, t4)
    for d in itertools.product((.5, -.5), repeat=3):
        small += H.vertex_cells(v + np.array(d), t, s, t4)
    cluster(big, small, v, "sesq_rd_honeycomb", az, el)


def honeycomb(az=28, el=22, explode=0.35):
    sys.path.insert(0, HERE)
    import sesq_honey as H
    t, s, t4 = np.load(os.path.join(HERE, "sesq_honey_params.npz"))["x"]
    v = np.array([1.0, 0, 0])
    cells = [H.sesq_cell(np.array(c, float), t, s, t4)
             for c in ([0, 0, 0], [2, 0, 0], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1])]
    cells += H.vertex_cells(v, t, s, t4)
    for d in itertools.product((.5, -.5), repeat=3):
        cells += H.vertex_cells(v + np.array(d), t, s, t4)
    X_all, faces_all, off = [], [], 0
    for P, F in cells:
        X_all.append(P + explode * (P.mean(0) - v))
        faces_all += [[i + off for i in f] for f in F]
        off += len(P)
    X = np.vstack(X_all) @ rot(az, el).T
    finish(draw(X, faces_all, alpha=1.0, light=True, lw=0.4, edge_depth=False), X[:, :2], "sesq_rd_honeycomb")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "24":
        polychoron("sesq24cell", 0)
    elif what == "jp":
        polychoron("sesqjp", 1, az=20, el=25)
    elif what == "honey":
        honeycomb_v2()
