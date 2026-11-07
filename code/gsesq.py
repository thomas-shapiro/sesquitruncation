r"""Generalized sesquitruncation: order-3 vertices truncated, a vertex of order k >= 4 replaced by
a k-gon (its flag corners) surrounded by k triangles.  Input: oriented face list of P.
    python3 gsesq.py   -> tests on the pentagonal pyramid etc.: types, sphericity, canonical ratio
"""
import os, sys, math
from collections import defaultdict, Counter
sys.path.insert(0, os.path.expanduser("~/nmj"))
from mapcheck import orient
from robust import canonical_robust


def gsesq(F):
    """F: oriented faces of P (lists of vertex ids). Returns faces of sigma(P) on new vertex ids."""
    nxt = {}                       # (face, vertex) -> next vertex along the face
    rot = defaultdict(list)        # vertex -> list of faces
    for fi, f in enumerate(F):
        for i, v in enumerate(f):
            nxt[(fi, v)] = f[(i + 1) % len(f)]
            rot[v].append(fi)
    order = {v: len(fs) for v, fs in rot.items()}
    ids = {}
    def vid(key):
        return ids.setdefault(key, len(ids))
    def dart(v, w):
        return vid(("d", v, w))
    def flag(v, fi):
        return vid(("f", v, fi))
    out = []
    for fi, f in enumerate(F):                      # big faces
        big = []
        m = len(f)
        for i, v in enumerate(f):
            prv, nx = f[i - 1], f[(i + 1) % m]
            big.append(dart(v, prv))
            if order[v] >= 4:
                big.append(flag(v, fi))
            big.append(dart(v, nx))
        out.append(big)
    # cyclic order of faces / neighbours around each vertex
    for v, fs in rot.items():
        # face containing directed edge v->w : the face with nxt[(fi, v)] == w; the next face around v
        # is the one containing w->v ... build cyclic sequence of neighbours
        out_of = {nxt[(fi, v)]: fi for fi in fs}                  # w -> face where v->w
        prev_of = {}
        for fi in fs:
            f = F[fi]; i = f.index(v); prev_of[fi] = f[i - 1]       # face fi has ... -> u -> v -> w
        # walk: start with some face fi; neighbours: w = nxt(fi,v); next face is the one where v->u with u = prev?
        fi = fs[0]; seq = []
        for _ in range(len(fs)):
            w = nxt[(fi, v)]
            seq.append((fi, w))
            # face on the other side of edge v-w containing w->v; in it, v -> next is the following neighbour
            g = [gj for gj in fs if prev_of[gj] == w][0]
            fi = g
        k = order[v]
        if k == 3:
            out.append([dart(v, w) for fi, w in seq][::-1])
        else:
            out.append([flag(v, fi) for fi, w in seq][::-1])
            for j, (fi, w) in enumerate(seq):
                g = seq[(j + 1) % k][0]      # the face on the other side of the edge v-w
                out.append([dart(v, w), flag(v, fi), flag(v, g)])
    return out, len(ids)


def types(G):
    at = defaultdict(list)
    for f in G:
        for v in f:
            at[v].append(len(f))
    return at


def spherical(G):
    worst = 0
    for v, ps in types(G).items():
        th = sum(180 - 360 / p for p in ps)
        worst = max(worst, th)
    return worst


def pyramid(k):
    base = list(range(k))
    apex = k
    F = [base[::-1]] + [[base[i], base[(i + 1) % k], apex] for i in range(k)]
    return F


if __name__ == "__main__":
    for name, F in (("square pyramid (check: = sigma, V20)", pyramid(4)), ("pentagonal pyramid", pyramid(5)),
                    ("hexagonal pyramid", pyramid(6))):
        F = orient(F)
        G, n = gsesq(F)
        G = orient(G)
        th = spherical(G)
        try:
            res = canonical_robust(G, n)
        except Exception:
            res = None
        print(name, "V", n, "faces", dict(sorted(Counter(len(f) for f in G).items())),
              "types", sorted({".".join(map(str, sorted(t))) for t in types(G).values()}),
              "max angle sum %.2f" % th, "ratio %.4f" % res[1]["ratio"] if res else "no canonical")
