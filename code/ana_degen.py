import sys, numpy as np
from collections import Counter, defaultdict
def read_off(fn):
    L = [l.split('#')[0].split() for l in open(fn)]
    L = [l for l in L if l]
    assert L[0][0].upper().endswith('OFF')
    nv, nf = int(L[1][0]), int(L[1][1])
    X = np.array([[float(x) for x in l[:3]] for l in L[2:2+nv]])
    F = [[int(x) for x in l[1:1+int(l[0])]] for l in L[2+nv:2+nv+nf]]
    return X, F
for fn in sys.argv[1:]:
    X, F = read_off(fn)
    E = {tuple(sorted((f[i], f[(i+1)%len(f)]))) for f in F for i in range(len(f))}
    Ls = np.array([np.linalg.norm(X[a]-X[b]) for a, b in E])
    # vertex types: cyclic face sizes
    at = defaultdict(list)
    for fi, f in enumerate(F):
        for i, v in enumerate(f): at[v].append((fi, f[i-1], f[(i+1)%len(f)]))
    types = Counter()
    for v, lst in at.items():
        # order faces cyclically: next face shares edge (v, nxt)
        by_prev = {p: (fi, n) for fi, p, n in lst}
        fi, p, n = lst[0]; seq = [len(F[fi])]; cur = n
        for _ in range(len(lst)-1):
            # face whose prev == ? find face containing edge (v,cur) other than fi
            cand = [x for x in lst if x[0] != fi and cur in (x[1], x[2])]
            fi, p2, n2 = cand[0]; seq.append(len(F[fi])); cur = n2 if p2 == cur else p2
        k = len(seq); rots = [tuple(seq[i:]+seq[:i]) for i in range(k)]; rots += [tuple(reversed(r)) for r in rots]
        t = min(rots)
        ang = sum(180*(m-2)/m for m in t)
        types[(t, "sph" if ang < 360-1e-9 else ("flat" if abs(ang-360) < 1e-9 else "hyp"))] += 1
    print(fn, "V", len(X), "E", len(E), "F", len(F), "faces", dict(sorted(Counter(len(f) for f in F).items())), "ratio %.5f" % (Ls.max()/Ls.min()))
    for (t, s), c in sorted(types.items()): print("   ", ".".join(map(str, t)), s, c)
    # centre / midsphere check
    c = X.mean(0); mids = np.array([np.linalg.norm((X[a]+X[b])/2 - c) for a, b in E])
    print("    midradius spread %.4f" % (mids.max()/mids.min()))
