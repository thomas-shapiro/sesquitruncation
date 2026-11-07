"""TRD honeycomb figure with the six-parameter fit of trd_fit.py (edge ratio ~1.17).
    python3 render_trd6.py az el [aniso]   (aniso: print the shape statistics only)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sesq_honey import hull_faces
from render_hd2 import COL, view, add_cell, tikz_faces
from render_hd import finish
HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("TRD7"):
    import trd_fit7 as T7
    T = T7.T
    y = np.load(os.path.join(HERE, "trd7_params.npy"))
    T7.setlam(y[-1]); x = y[:-1]
else:
    import trd_fit as T
    x = np.load(os.path.join(HERE, "trd_params.npy"))
nd0, nf0 = len(T.dkeys), len(T.fkeys)
dart, flag = T.pts(x)


def patch(Rad=2.2):
    out = []
    keep = [p for p in T.CF if np.linalg.norm(T.X[p]) <= Rad]
    for p in keep:
        P = []
        for f in T.CF[p]:
            for i, v in enumerate(f):
                P += [dart(v, f[i - 1]), dart(v, f[(i + 1) % len(f)])]
                if v in T.cub:
                    P.append(flag(v, f))
        P = np.unique(np.round(np.array(P), 9), axis=0)
        out.append((P, hull_faces(P), "big"))
    vs = {v for p in keep for f in T.CF[p] for v in f}
    for v in vs:
        if len(T.nb[v]) not in (4, 8) or len(T.faces_at[v]) < (6 if len(T.nb[v]) == 4 else 12):
            continue
        if v not in T.cub:
            P = np.array([dart(v, w) for w in T.nb[v]]); out.append((P, hull_faces(P), "tetS"))
            continue
        fl = {frozenset((f[f.index(v) - 1], f[(f.index(v) + 1) % len(f)])): flag(v, f) for f in T.faces_at[v]}
        Q = np.array(list(fl.values())); out.append((Q, hull_faces(Q), "cuboct"))
        for w in T.nb[v]:
            Tt = np.array([dart(v, w)] + [p for k, p in fl.items() if w in k]); out.append((Tt, hull_faces(Tt), "tetC"))
    assert (len(T.dkeys), len(T.fkeys)) == (nd0, nf0), "new parameter classes appeared"
    return out


if __name__ == "__main__":
    az, el = float(sys.argv[1]), float(sys.argv[2])
    cells = patch()
    if len(sys.argv) > 3:
        from collections import defaultdict
        acc = defaultdict(lambda: np.zeros((3, 3))); cnt = defaultdict(int); L = []
        for P, F, kd in cells:
            Q = P - P.mean(0); C = Q.T @ Q; acc[kd] += C / np.trace(C); cnt[kd] += 1
            for f in F:
                for i in range(len(f)): L.append(np.linalg.norm(P[f[i]] - P[f[i - 1]]))
        for k in acc:
            C = acc[k] / cnt[k]; print(k, cnt[k], "zz/xx %.3f yy/xx %.3f" % (C[2, 2] / C[0, 0], C[1, 1] / C[0, 0]))
        print("edge ratio %.4f" % (max(L) / min(L)))
        sys.exit()
    R = view(az, el)
    # 2D correction: scale the picture vertically so that, averaged over all cells (weighted by
    # their projected size), the cells are as tall as they are wide
    num = den = 0.0
    for P, F, kd in cells:
        Q = P @ R.T; Q = Q[:, :2] - Q[:, :2].mean(0)
        sx, sy = Q[:, 0].std(), Q[:, 1].std(); w = sx * sy
        num += w * np.log(sy / sx); den += w
    k = np.exp(-num / den)
    print("vertical 2D scale %.4f" % k)
    R = np.diag([1.0, k, 1.0]) @ R
    items = []
    for P, F, kd in cells:
        add_cell(items, P, F, COL[kd], R)
    finish(tikz_faces(items), np.vstack([it[1] for it in items]), "sesq_trd_honeycomb", width_cm=10)
