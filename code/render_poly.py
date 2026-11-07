r"""Shaded polyhedron renders for the paper (TikZ -> PDF, same look as render_hd2).
Face colours follow the size scheme 3 yellow, 4 red, 5 blue, 6 magenta, odd >6 green, even >6 orange.
    python3 render_poly.py <file.off> <name> [az el] [--coords file.npy] [--axis k]
--axis k: look down the normal of a face with k sides (centred symmetric view)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_hd import rot, finish
from ana_degen_io import read_off

def colour(n):
    if n == 3: return (0.95, 0.85, 0.20)
    if n == 4: return (0.85, 0.25, 0.22)
    if n == 5: return (0.30, 0.42, 0.78)
    if n == 6: return (0.78, 0.35, 0.72)
    if n >= 12:                                       # darker tones for 12 sides and more
        return (0.15, 0.50, 0.50) if n % 2 else (0.78, 0.36, 0.12)
    return (0.40, 0.72, 0.32) if n % 2 else (0.95, 0.58, 0.18)

L = np.array([-0.35, 0.55, 0.76]); L /= np.linalg.norm(L)

def render(V, F, name, az=20, el=18, axis=None, width_cm=5.0, lw=0.3):
    V = V - V.mean(0)
    R = rot(az, el)
    if axis:
        f = max((f for f in F if len(f) == axis), key=lambda f: np.linalg.norm(V[f].mean(0)))
        n = V[f].mean(0); n /= np.linalg.norm(n)
        a = np.cross(n, [0, 0, 1.0]); sa = np.linalg.norm(a); ca = n[2]
        if sa > 1e-9:
            a /= sa
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = rot(az, el) @ (np.eye(3) + sa * K + (1 - ca) * K @ K)
        elif ca < 0:
            R = rot(az, el) @ np.diag([1.0, -1.0, -1.0])
    X = V @ R.T
    items, cols = [], {}
    for f in F:
        P = X[f]; c = P.mean(0)
        n = np.linalg.svd(P - c)[2][2]
        if n @ c < 0: n = -n
        if n[2] <= 1e-9: continue                      # back face
        sh = 0.55 + 0.45 * max(0.0, n @ L)
        rgb = tuple(min(1, x * sh + 0.10) for x in colour(len(f)))
        key = "%.3f,%.3f,%.3f" % rgb
        cols.setdefault(key, "c%d" % len(cols))
        items.append((c[2], P[:, :2], cols[key]))
    items.sort(key=lambda it: it[0])
    out = ["\\definecolor{%s}{rgb}{%s}" % (v, k) for k, v in cols.items()]
    for z, P2, cn in items:
        pts = " -- ".join("(%.4f,%.4f)" % (x, y) for x, y in P2)
        out.append("\\filldraw[fill=%s,draw=black!85,line width=%.2fpt,line join=round] %s -- cycle;" % (cn, lw, pts))
    finish(out, X[:, :2], name, width_cm=width_cm)

if __name__ == "__main__":
    a = sys.argv[1:]
    coords = None; axis = None
    if "--coords" in a:
        i = a.index("--coords"); coords = a[i + 1]; del a[i:i + 2]
    if "--axis" in a:
        i = a.index("--axis"); axis = int(a[i + 1]); del a[i:i + 2]
    V, F = read_off(a[0])
    if coords: V = np.load(coords)
    az, el = (float(a[2]), float(a[3])) if len(a) > 3 else (20, 18)
    render(V, F, a[1], az, el, axis)
