r"""Compare generator 2 (gen2/n<N>.txt) with the typed-map family (ext2/family_ids.json), per parent size.
    python3 gen2_compare.py 4 10
"""
import json, os, sys
from itertools import combinations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen2_plantri import canon

fam = json.load(open(os.path.expanduser("~/nmj/sesqpaper/ext2/family_ids.json")))


def rotation(F, n):
    nxt = [dict() for _ in range(n)]
    for f in F:
        k = len(f)
        for j in range(k):
            a, v, b = f[j - 1], f[j], f[(j + 1) % k]
            nxt[v][b] = a
    adj = []
    for v in range(n):
        if not nxt[v]:
            return None
        s = next(iter(nxt[v])); r = [s]
        while nxt[v][r[-1]] != s:
            r.append(nxt[v][r[-1]])
        if len(r) != len(nxt[v]):
            return None
        adj.append(r)
    return adj


def three_connected(adj):
    n = len(adj)
    if any(len(set(a)) != len(a) or len(a) < 3 for a in adj):
        return False
    for x, y in combinations(range(n), 2):
        rest = [v for v in range(n) if v not in (x, y)]
        seen = {rest[0]}; st = [rest[0]]
        while st:
            u = st.pop()
            for w in adj[u]:
                if w not in seen and w not in (x, y):
                    seen.add(w); st.append(w)
        if len(seen) != len(rest):
            return False
    return True


lo, hi = int(sys.argv[1]), int(sys.argv[2])
ok_all = True
for n in range(lo, hi + 1):
    g2 = {eval(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen2", "n%d.txt" % n))}
    g1, degen = {}, 0
    for x in fam:
        if x["pnv"] != n:
            continue
        adj = rotation(x["parent"], n)
        if adj is None or not three_connected(adj):
            degen += 1; continue
        g1[canon(adj)] = x["id"]
    only1 = [g1[c] for c in g1 if c not in g2]
    only2 = len(g2 - set(g1))
    same = len(g1) == len(set(g1.values()))
    print("n=%2d gen1=%4d (+%d degenerate) gen2=%4d only_gen1=%d only_gen2=%d %s" %
          (n, len(g1), degen, len(g2), len(only1), only2, "MATCH" if not only1 and not only2 else "DIFF " + str(only1[:5])))
    ok_all &= not only1 and not only2
print("ALL MATCH" if ok_all else "MISMATCH")
