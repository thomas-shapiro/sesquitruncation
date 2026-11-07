r"""Generator 2 (independent check): plantri -pc3f4 n lists every 3-connected planar map with
triangular and quadrilateral faces on n vertices (= dual of an input polyhedron P with vertex
orders 3/4 and n faces). For each, build the sesquitruncate directly and keep it iff every
vertex is strictly spherical (angle sum of regular polygons < 2pi). Output canonical codes.
    plantri -pc3f4 n | python3 gen2_plantri.py n > gen2_n.txt
Shares no code with the typed-map generator (tmap3x / ext_gen).
"""
import sys
from math import pi

ANG = {k: pi * (k - 2) / k for k in range(3, 64)}


def read_planar_code(stream):
    hdr = stream.read(15)
    assert hdr == b">>planar_code<<", hdr
    while True:
        b = stream.read(1)
        if not b:
            return
        n = b[0]
        adj, cur = [], []
        while len(adj) < n:
            x = stream.read(1)[0]
            if x == 0:
                adj.append(cur); cur = []
            else:
                cur.append(x - 1)
        yield adj


def faces_of(adj):
    """faces from a rotation system (plantri: neighbours in clockwise order)"""
    pos = [{w: i for i, w in enumerate(a)} for a in adj]
    seen, F = set(), []
    for u in range(len(adj)):
        for v in adj[u]:
            if (u, v) in seen:
                continue
            f, a, b = [], u, v
            while (a, b) not in seen:
                seen.add((a, b)); f.append(a)
                i = pos[b][a]
                a, b = b, adj[b][(i - 1) % len(adj[b])]
            F.append(f)
    return F


def sesq_ok(adj, F):
    """parent vertex phi (= face of P) with t triangles and s squares around it -> big face 2t+3s.
    child vertices: x on each P-edge end: type 3.n.m (n, m big faces either side of the P-edge);
    y in each order-4 corner: 3.4.3.n. Strictly spherical test on each."""
    big = [0] * len(adj)
    for f in F:
        for v in f:
            big[v] += 2 if len(f) == 3 else 3
    for u in range(len(adj)):
        for v in adj[u]:
            if u < v and ANG[3] + ANG[big[u]] + ANG[big[v]] >= 2 * pi - 1e-12:
                return False
    for f in F:
        if len(f) == 4:
            for v in f:
                if 2 * ANG[3] + ANG[4] + ANG[big[v]] >= 2 * pi - 1e-12:
                    return False
    return True


def canon(adj):
    """canonical code of the map up to orientation-preserving and -reversing isomorphism"""
    best = None
    n = len(adj)
    pos = [{w: i for i, w in enumerate(a)} for a in adj]
    for d in (1, -1):
        for u in range(n):
            for v in adj[u]:
                lab = {u: 0}; order = [u]; first = {u: v}; code = []
                i = 0
                while i < len(order):
                    a = order[i]; i += 1
                    k = len(adj[a]); j0 = pos[a][first[a]]
                    row = []
                    for t in range(k):
                        w = adj[a][(j0 + d * t) % k]
                        if w not in lab:
                            lab[w] = len(order); order.append(w); first[w] = a
                        row.append(lab[w])
                    code.append(tuple(row))
                code = tuple(code)
                if best is None or code < best:
                    best = code
    return best


if __name__ == "__main__":
    n0 = 0
    for adj in read_planar_code(sys.stdin.buffer):
        n0 += 1
        if max(len(a) for a in adj) > 11:
            continue
        F = faces_of(adj)
        if sesq_ok(adj, F):
            print(repr(canon(adj)))
    print("# read %d" % n0, file=sys.stderr)
