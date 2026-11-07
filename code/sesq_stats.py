r"""Per-member statistics for the sesquitruncation paper: combinatorial symmetry group
(= point group of the canonical form for a 3-connected map), V, F, big faces, V3/V4 of the input.
    python3 sesq_stats.py -> ~/nmj/sesqpaper/ext2/stats.json
"""
import json, os, sys
from collections import Counter
from math import gcd
sys.path.insert(0, os.path.expanduser("~/nmj"))
from mapcheck import orient

SQ = os.path.expanduser("~/nmj/sesqpaper/ext2")


def darts(F):
    phi, left = {}, {}
    for fi, f in enumerate(F):
        k = len(f)
        for j in range(k):
            a, b, c = f[j], f[(j + 1) % k], f[(j + 2) % k]
            phi[(a, b)] = (b, c); left[(a, b)] = fi
    return phi, left


def isos(F1, F2, key1, key2):
    """all orientation-preserving map isomorphisms F1 -> F2, as vertex permutations"""
    phi1, _ = darts(F1); phi2, _ = darts(F2)
    D1 = list(phi1)
    k0 = Counter(key1[d] for d in D1)
    kmin = min(k0, key=lambda k: (k0[k], k))
    d0 = next(d for d in D1 if key1[d] == kmin)
    out = []
    for e0 in phi2:
        if key2[e0] != kmin:
            continue
        m = {d0: e0}; st = [d0]; ok = True
        while st and ok:
            d = st.pop(); e = m[d]
            for dd, ee in ((phi1[d], phi2[e]), ((d[1], d[0]), (e[1], e[0]))):
                if dd in m:
                    if m[dd] != ee: ok = False; break
                else:
                    if ee not in phi2 or key1[dd] != key2[ee]: ok = False; break
                    m[dd] = ee; st.append(dd)
        if ok and len(m) == len(D1):
            vp = {}
            for d, e in m.items():
                if vp.setdefault(d[0], e[0]) != e[0]: ok = False; break
            if ok:
                out.append(vp)
    return out


def dkey(F):
    phi, left = darts(F)
    L = {d: len(F[left[d]]) for d in phi}
    return {d: (L[d], L[(d[1], d[0])]) for d in phi}


def porder(p):
    seen, o = set(), 1
    for v in p:
        if v in seen: continue
        n, w = 0, v
        while w not in seen:
            seen.add(w); w = p[w]; n += 1
        o = o * n // gcd(o, n)
    return o


def fixes_cell(p, F, E):
    if any(p[v] == v for v in p): return True
    if any({p[a], p[b]} == {a, b} for a, b in E): return True
    S = {frozenset(f) for f in F}
    return any(frozenset(p[v] for v in f) == frozenset(f) for f in F)


def group(F):
    F = orient([list(f) for f in F])
    Fm = [list(reversed(f)) for f in F]
    rot = isos(F, F, dkey(F), dkey(F))
    refl = isos(F, Fm, dkey(F), dkey(Fm))
    n = len(rot)
    ords = Counter(porder(p) for p in rot)
    E = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in F for i in range(len(f))}
    if n == 60: R, m = "I", 0
    elif n == 24 and ords[4]: R, m = "O", 0
    elif n == 12 and ords[3] == 8 and not ords[6] and not ords[12]: R, m = "T", 0
    elif ords[n]: R, m = "C", n
    else: R, m = "D", n // 2
    if not refl:
        return {"C": "C_{%d}" % m, "D": "D_{%d}" % m}.get(R, R), 2 * n if refl else n
    mirrors = sum(1 for p in refl if porder(p) == 2 and fixes_cell(p, F, E))
    if R == "I": g = "I_h"
    elif R == "O": g = "O_h"
    elif R == "T": g = "T_d" if mirrors == 6 else "T_h"
    elif R == "C":
        if m == 1: g = "C_s" if mirrors else "C_i"
        elif mirrors == m: g = "C_{%dv}" % m
        elif mirrors == 1: g = "C_{%dh}" % m
        else: g = "S_{%d}" % (2 * m)
    else:
        g = ("D_{%dh}" % m) if mirrors == m + 1 else ("D_{%dd}" % m)
    if R == "C" and m == 2 and mirrors == 1: g = "C_{2h}"
    return g, 2 * n


if __name__ == "__main__":
    fam = json.load(open(os.path.join(SQ, "family_ids.json")))
    rows = []
    for x in fam:
        g, order = group(x["child"])
        P = x["parent"]
        rows.append({"id": x["id"], "V": x["cnv"], "F": len(x["child"]), "big": x["big"],
                     "V3": sum(len(f) == 3 for f in P), "V4": sum(len(f) == 4 for f in P),
                     "inF": x["pnv"], "canon": x["canon"], "group": g, "order": order,
                     "faces": dict(sorted(Counter(len(f) for f in x["child"]).items()))})
        if len(rows) % 500 == 0: print(len(rows), flush=True)
    json.dump(rows, open(os.path.join(SQ, "stats.json"), "w"))
    print("done", len(rows))
