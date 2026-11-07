"""fit the sesquitruncated joined pentachoron (base polytope fixed).
params: t  = dart fraction at the simplicial end of an s-c edge,
        a  = dart fraction at the cubical end of an s-c edge,
        b  = dart fraction at either end of a c-c edge,
        s  = flag hyperplane depth.
Valid = convex position, 60 cells, 120 flag-flag edges, no dart-dart edge at a cubical vertex."""
import sys, numpy as np
from collections import Counter
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sesq4d as S

V, E, F, C, cub = S.show("jp", S.joined_pentachoron())


def make(x):
    t, a, b, s, al = x if len(x) == 5 else (*x, 0.5)
    P, lab = S.sesq_points(V, E, F, t, s, cub, a)
    P = P.copy()
    # flags: direction to al*w_s + (1-al)*w_c, all flags of v on one hyperplane orthogonal to v
    by_v = {}
    for i, l in enumerate(lab):
        if l[0] == "flag":
            v, k = l[1], l[2]
            f = F[k]; j = f.index(v)
            nbs = [f[j - 1], f[(j + 1) % len(f)]]
            ws = [w for w in nbs if w not in cub]; wc = [w for w in nbs if w in cub]
            m = al * V[ws[0]] + (1 - al) * V[wc[0]] if ws and wc else V[nbs].mean(0)
            by_v.setdefault(v, []).append((i, m))
    for v, ms in by_v.items():
        u = V[v] / np.linalg.norm(V[v])
        h = s * min((V[v] - m) @ u for i, m in ms)
        for i, m in ms:
            P[i] = V[v] + h / ((V[v] - m) @ u) * (m - V[v])
    for i, l in enumerate(lab):
        if l[0] == "dart":
            v = l[1]; u, w = l[2]; w = w if u == v else u
            if v in cub and w in cub:
                P[i] = V[v] + b * (V[w] - V[v])
            elif v in cub:
                P[i] = V[v] + a * (V[w] - V[v])
            else:
                P[i] = V[v] + t * (V[w] - V[v])
    return P, lab


def score(x, verbose=False):
    t, a, b, s, al = x
    if not (0.05 < al < 0.95 and 0.02 < t < 0.7 and 0.02 < a < 0.7 and 0.02 < b < 0.49 and t + a < 0.97 and 0.03 < s < 2.0):
        return 99.0
    P, lab = make(x)
    try:
        if len(ConvexHull(P).vertices) != len(P):
            return 50.0
        E2, F2, C2 = S.faces4(P)
    except Exception:
        return 99.0
    cnt = Counter()
    for p, q in E2:
        lp, lq = lab[p], lab[q]
        if lp[0] == lq[0] == "flag":
            cnt["ff"] += 1
        if lp[0] == lq[0] == "dart" and lp[1] == lq[1] and lp[1] in cub:
            cnt["bad"] += 1
    if verbose:
        print("cells", len(C2), dict(cnt))
    if len(C2) != 60 or cnt["ff"] != 120 or cnt["bad"]:
        return 20.0 + abs(len(C2) - 60) + abs(cnt["ff"] - 120) / 10 + cnt["bad"] / 10
    L = np.array([np.linalg.norm(P[p] - P[q]) for p, q in E2])
    return L.max() / L.min()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x0 = np.array([0.13831, 0.57615, 0.4303, 0.3877, 0.5])
    best = (score(x0), x0)
    print('seed', best[0], flush=True)
    for k in range(int(sys.argv[1]) if len(sys.argv) > 1 else 300):
        x = np.array([rng.uniform(0.05, 0.6), rng.uniform(0.05, 0.6), rng.uniform(0.05, 0.48), rng.uniform(0.05, 1.5), rng.uniform(0.2, 0.8)])
        r = score(x)
        if r < best[0]:
            best = (r, x)
            print("sample %d score %.4f" % (k, r), np.round(x, 4), flush=True)
    for k in range(10):
        x0 = best[1] if k == 0 else best[1] * (1 + rng.normal(0, 0.05, 5))
        o = minimize(score, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 1000})
        if o.fun < best[0]:
            best = (o.fun, o.x)
            print("NM %d ratio %.5f" % (k, o.fun), np.round(o.x, 5), flush=True)
    score(best[1], verbose=True)
    P, lab = make(best[1])
    np.save("sesq4d_1.npy", P)
    np.save("jp_params.npy", best[1])
    print("final ratio %.5f" % best[0], np.round(best[1], 5))
