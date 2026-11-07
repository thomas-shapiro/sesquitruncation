r"""Best edge ratio over all MIDSCRIBED realizations of each proper member.

All realizations with every edge tangent to the unit sphere are images of the canonical
one under Moebius transformations of the sphere (Koebe-Andreev-Thurston / Springborn).
They keep every face a tangential polygon (an incircle), so faces stay "round"; the
canonical one is the one whose tangency points have barycentre 0.
For each member: canonical tangency points t_e -> phi_a(t_e) for a in the open ball,
vertex u' = pole of the plane through its tangency points, minimise the edge ratio over a.

    python3 mobius.py [nproc]   -> ~/nmj/sesqpaper/ext2/mobius.json  (resumable)
"""
import json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
sys.path.insert(0, os.path.expanduser("~/nmj"))
from robust import canonical_robust
from mapcheck import orient

SQ = os.path.expanduser("~/nmj/sesqpaper/ext2")
OUTF = os.path.join(SQ, "mobius.json")


def phi(a, X):
    """Moebius automorphism of the unit ball sending a to 0 (applied to unit vectors X)."""
    aa = a @ a
    d = X - a
    num = (1 - aa) * d - (d * d).sum(1)[:, None] * a
    den = 1 - 2 * X @ a + aa * (X * X).sum(1)
    return num / den[:, None]


def setup(F, nv, V):
    E = sorted({(min(f[i], f[(i + 1) % len(f)]), max(f[i], f[(i + 1) % len(f)])) for f in F for i in range(len(f))})
    E = np.array(E)
    A, B = V[E[:, 0]], V[E[:, 1]]
    D = B - A
    s = -(A * D).sum(1) / (D * D).sum(1)
    T = A + s[:, None] * D                       # foot of the perpendicular = tangency point
    T /= np.linalg.norm(T, axis=1)[:, None]
    inc = [[] for _ in range(nv)]
    for k, (u, w) in enumerate(E):
        inc[u].append(k); inc[w].append(k)
    return E, T, inc


def realize(T, inc):
    P = np.empty((len(inc), 3))
    for u, ks in enumerate(inc):
        X = T[ks]
        c = X.mean(0)
        n = np.linalg.svd(X - c)[2][-1]
        d = n @ c
        if d < 0:
            n, d = -n, -d
        P[u] = n / d
    return P


def ratio(P, E):
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    return L.max() / L.min()


def work(x):
    F = orient([list(f) for f in x["child"]])
    nv = x["cnv"]
    res = canonical_robust(F, nv)
    if res is None:
        return x["id"], None
    V = np.asarray(res[0], float)
    E, T, inc = setup(F, nv, V)
    r0 = ratio(realize(T, inc), E)

    def f(w):
        n = np.linalg.norm(w)
        a = np.tanh(n) * w / n if n > 1e-12 else np.zeros(3)
        return np.log(ratio(realize(phi(a, T), inc), E))

    best = (np.log(r0), np.zeros(3))
    rng = np.random.default_rng(0)
    for k in range(6):
        w0 = np.zeros(3) if k == 0 else rng.normal(0, 0.4, 3)
        o = minimize(f, w0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
        if o.fun < best[0]:
            best = (o.fun, o.x)
    w = best[1]
    n = np.linalg.norm(w)
    return x["id"], {"canon": float(r0), "mob": float(np.exp(best[0])), "dist": float(2 * n),
                     "a": [float(v) for v in (np.tanh(n) * w / n if n > 1e-12 else np.zeros(3))]}


if __name__ == "__main__":
    fam = json.load(open(os.path.join(SQ, "family_ids.json")))
    stats = {r["id"]: r for r in json.load(open(os.path.join(SQ, "stats.json")))}
    deg = set(json.load(open(os.path.join(SQ, "degenerate_ids.json"))))
    done = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
    todo = [x for x in fam if x["id"] not in deg and stats[x["id"]]["V4"] > 0 and x["id"] not in done and stats[x["id"]]["canon"] < float(os.environ.get("RHOMAX", "9"))]
    todo.sort(key=lambda x: stats[x["id"]]["canon"])
    print("todo", len(todo), flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(work, todo, chunksize=4)):
            done[i] = r
            if r and r["mob"] < 1.1:
                print("%s canon=%.4f mob=%.4f dist=%.3f" % (i, r["canon"], r["mob"], r["dist"]), flush=True)
            if k % 200 == 0:
                json.dump(done, open(OUTF, "w"))
                print("..%d" % k, flush=True)
    json.dump(done, open(OUTF, "w"))
    print("done", len(done))
