"""Exhaustive generator of convex-polyhedral maps over a fixed vocabulary of
CYCLIC vertex types.  Rewrite of tmap.py with a real constraint engine.

State
-----
Each vertex v carries fs[v], its corner face-sizes in rotation order, and
nb[v], the neighbour in each edge slot (-1 = open).  Slot i of v lies between
corner i-1 and corner i, so filling one slot merges two partial faces.

Partial faces are tracked by a union-find over CORNERS with rollback.  Linking
slot i of v to slot j of u performs exactly two unions:

    corner (v,i)   -> corner (u,j-1)      (the face on one side of the edge)
    corner (u,j)   -> corner (v,i-1)      (the face on the other side)

Every union joins corners whose target face size already agrees, so a chain's
target size is well defined.  A chain of target size k is illegal once it
exceeds k corners, and when it reaches exactly k its two ends are forced to
join - that closure is applied by unit propagation, not by branching, which is
where nearly all of the pruning comes from.

Exhaustiveness
--------------
Descartes bounds the vertex count: V <= 720 / (smallest defect in the
vocabulary), and the running defect sum is pruned against 720 exactly (in
integer units of 1/2520 degree).  So a completed run is a genuine proof, not a
budget cutoff; `hitcap` reports whether the node limit was reached instead.
"""
import sys, os, time, json, math
from collections import Counter
sys.path.insert(0, os.path.expanduser('~/nmj'))


def angle(k):
    return 180.0 - 360.0 / k


def defect(seq):
    return 360.0 - sum(angle(k) for k in seq)


def rotations(seq):
    seq = tuple(seq)
    return {seq[i:] + seq[:i] for i in range(len(seq))}


def canon_cyclic(seq):
    """smallest rotation of seq or of its reverse - the identity of a type"""
    n = len(seq)
    return min([s[i:] + s[:i]
                for s in (tuple(seq), tuple(reversed(seq)))
                for i in range(n)])


class Gen:
    def __init__(self, types, maxV=None, need=None, verbose=False,
                 maxnodes=float('inf'), onfind=None, maxface=None,
                 weights=None, total=None):
        """`weights`/`total` replace the angular-defect budget with any other
        conserved per-vertex quantity.  Sesquitruncation parents need this: they
        are plain combinatorial maps with no Descartes bound, but their child's
        big faces obey  sum over vertices of (12 - n_v) = 24, which bounds them
        just as tightly."""
        base = set()
        for t in types:
            base |= rotations(t)
            base |= rotations(tuple(reversed(t)))
        self.allfs = sorted(base)
        # Descartes must be EXACT integer arithmetic.  Scaling by a fixed 2520
        # silently breaks for any face size that does not divide it (11, 13, ...),
        # so the denominator is the lcm of the face sizes actually in play:
        #   defect * L = 360L - 180*d*L + 360*sum(L/k)
        if weights is not None:
            self.dfx = {}
            for fs in self.allfs:
                key = canon_cyclic(fs)
                self.dfx[fs] = weights[key] if key in weights else weights[fs]
            self.TOT = total
            wmin = min(w for w in self.dfx.values() if w > 0) if any(
                w > 0 for w in self.dfx.values()) else None
            self.maxV = maxV or (total // wmin if wmin else 10 ** 6)
        else:
            sizes = {k for fs in self.allfs for k in fs}
            L = 1
            for k in sizes:
                L = L * k // math.gcd(L, k)
            self.L = L
            self.dfx = {fs: 360 * L - 180 * len(fs) * L + 360 * sum(L // k for k in fs)
                        for fs in self.allfs}
            self.TOT = 720 * L
            assert all(d > 0 for d in self.dfx.values()), 'non-positive defect'
            dmin = min(defect(t) for t in types)
            self.maxV = maxV or int(720.0 / dmin + 1e-9)
        self.ckey = {fs: canon_cyclic(fs) for fs in self.allfs}
        # new vertex attached by its slot 0: needs corner d-1 on face k, corner 0 on face g
        self.newcands = {}
        for fs in self.allfs:
            self.newcands.setdefault((fs[len(fs) - 1], fs[0]), []).append(fs)
        self.need = None
        if need is not None:
            self.need = {canon_cyclic(k): v for k, v in need.items()}
            self.maxV = min(self.maxV, sum(self.need.values()))
        self.verbose = verbose
        self.maxnodes = maxnodes
        self.onfind = onfind
        self.found = {}
        self.nfound = 0
        self.nodes = 0
        self.hitcap = False
        self.types0 = sorted({tuple(t) for t in types})
        self.t0 = time.time()
        self.prefix = ()          # fixed choice indices replayed before exploring
        self.split_depth = None   # when set, collect subtree prefixes instead of searching
        self.tasks = []
        self.path = [0] * 4096
        self.seedfilter = None    # restrict which seed type the search starts from
        self.tidx = {t: i for i, t in enumerate(self.types0)}
        self.curmin = 0           # every map has a unique MINIMUM vertex type;
                                  # only build it in the pass seeded by that type,
                                  # so a map is generated once instead of once per
                                  # distinct type it contains
        self.forbid = set()       # unordered pairs of canonical types that may
                                  # not share an edge (e.g. two sesq dodecagon
                                  # vertices, whose child vertex 3.12.12 is flat)

    # -------------------------------------------------------------- structure
    def reset(self, t0):
        self.fs = []
        self.dg = []
        self.nb = []
        self.cbase = []
        self.par = []      # union-find parent over corners
        self.sz = []       # chain length
        self.ktg = []      # target face size of the chain
        self.hd = []       # head corner of the chain
        self.tl = []       # tail corner of the chain
        self.cl = []       # chain already closed into a cycle
        self.trail = []
        self.dsum = 0
        self.tcount = Counter()
        self.forced = []
        self.add_vertex(tuple(t0))

    def add_vertex(self, fsw):
        v = len(self.fs)
        d = len(fsw)
        b = len(self.par)
        self.fs.append(fsw)
        self.dg.append(d)
        self.nb.append([-1] * d)
        self.cbase.append(b)
        for i in range(d):
            self.par.append(b + i)
            self.sz.append(1)
            self.ktg.append(fsw[i])
            self.hd.append(b + i)
            self.tl.append(b + i)
            self.cl.append(False)
        self.dsum += self.dfx[fsw]
        self.tcount[self.ckey[fsw]] += 1
        self.trail.append(('V', v, fsw))
        return v

    def find(self, x):
        p = self.par
        while p[x] != x:
            x = p[x]
        return x

    def mark(self):
        return len(self.trail)

    def undo(self, m):
        tr = self.trail
        while len(tr) > m:
            rec = tr.pop()
            k = rec[0]
            if k == 'V':
                _, v, fsw = rec
                d = len(fsw)
                b = self.cbase.pop()
                del self.par[b:]; del self.sz[b:]; del self.ktg[b:]
                del self.hd[b:]; del self.tl[b:]; del self.cl[b:]
                self.fs.pop(); self.dg.pop(); self.nb.pop()
                self.dsum -= self.dfx[fsw]
                self.tcount[self.ckey[fsw]] -= 1
            elif k == 'L':
                _, v, i, u, j = rec
                self.nb[v][i] = -1
                self.nb[u][j] = -1
            elif k == 'U':
                _, child, root, osz, ohd, otl = rec
                self.par[child] = child
                self.sz[root] = osz
                self.hd[root] = ohd
                self.tl[root] = otl
            else:  # 'C'
                self.cl[rec[1]] = False

    # ------------------------------------------------------------- propagation
    def join(self, ca, cb):
        """ca is the tail-side corner, cb the head-side corner of the next chain"""
        ra, rb = self.find(ca), self.find(cb)
        if ra == rb:
            # the chain has closed into a cycle
            if self.cl[ra] or self.sz[ra] != self.ktg[ra]:
                return False
            self.cl[ra] = True
            self.trail.append(('C', ra))
            return True
        if self.cl[ra] or self.cl[rb]:
            return False
        k = self.ktg[ra]
        n = self.sz[ra] + self.sz[rb]
        if n > k:
            return False
        if self.sz[ra] >= self.sz[rb]:
            keep, drop = ra, rb
        else:
            keep, drop = rb, ra
        self.trail.append(('U', drop, keep, self.sz[keep], self.hd[keep], self.tl[keep]))
        self.par[drop] = keep
        self.sz[keep] = n
        self.hd[keep] = self.hd[ra]
        self.tl[keep] = self.tl[rb]
        if n == k:
            self.forced.append(keep)
        return True

    def corner(self, v, i):
        return self.cbase[v] + i

    def link(self, v, i, u, j):
        """join slot i of v to slot j of u; False if that is inconsistent"""
        nb, fs = self.nb, self.fs
        if u == v or nb[v][i] != -1 or nb[u][j] != -1 or u in nb[v]:
            return False
        d, du = self.dg[v], self.dg[u]
        if fs[u][(j - 1) % du] != fs[v][i] or fs[u][j] != fs[v][(i - 1) % d]:
            return False
        if self.forbid:
            a, b = self.ckey[fs[v]], self.ckey[fs[u]]
            if (a, b) in self.forbid or (b, a) in self.forbid:
                return False
        nb[v][i] = u
        nb[u][j] = v
        self.trail.append(('L', v, i, u, j))
        return (self.join(self.corner(v, i), self.corner(u, (j - 1) % du)) and
                self.join(self.corner(u, j), self.corner(v, (i - 1) % d)))

    def propagate(self):
        """close every chain that has reached its full size; no branching"""
        while self.forced:
            r = self.forced.pop()
            r = self.find(r)
            if self.cl[r] or self.sz[r] != self.ktg[r]:
                continue
            a, b = self.tl[r], self.hd[r]        # a's next slot and b's previous slot
            va, ia = self.owner(a)
            vb, ib = self.owner(b)
            db = self.dg[vb]
            jb = (ib + 1) % db
            if self.nb[va][ia] != -1 or self.nb[vb][jb] != -1:
                return False
            if not self.link(va, ia, vb, jb):
                return False
        return True

    def owner(self, c):
        """(vertex, corner index) of corner id c"""
        cb = self.cbase
        lo, hi = 0, len(cb) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cb[mid] <= c:
                lo = mid
            else:
                hi = mid - 1
        return lo, c - cb[lo]

    # ------------------------------------------------------------------ search
    def pick(self):
        """open slot on the most nearly complete face; None if the map is full"""
        best, bestgap = None, 1 << 30
        for v in range(len(self.fs)):
            nbv = self.nb[v]
            if -1 not in nbv:
                continue
            for i in range(self.dg[v]):
                if nbv[i] == -1:
                    r = self.find(self.corner(v, i))
                    gap = self.ktg[r] - self.sz[r]
                    if gap < bestgap:
                        best, bestgap = (v, i), gap
                        if gap <= 1:
                            return best
        return best

    def run(self):
        for si, t in enumerate(self.types0):
            if self.seedfilter is not None and si != self.seedfilter:
                continue
            self.curmin = si
            self.reset(t)
            m = self.mark()
            self.dfs()
            self.undo(m)
            if self.hitcap:
                break
        return self.found

    def dfs(self, depth=0):
        """depth counts BRANCH points, so a choice path (one index per branch
        point) names a subtree exactly.  That is what lets one vocabulary be
        split across many cores: a worker replays a fixed prefix of choices and
        explores only the subtree underneath it, and the prefixes partition the
        whole search."""
        self.nodes += 1
        if self.nodes >= self.maxnodes:
            self.hitcap = True
            return
        if self.verbose and self.nodes % 500000 == 0:
            print(f'  ... {self.nodes} nodes  V={len(self.fs)}  found={len(self.found)}'
                  f'  {time.time()-self.t0:.0f}s', flush=True)
        savedF = self.forced
        self.forced = list(savedF)
        if not self.propagate():
            self.forced = savedF
            return
        p = self.pick()
        if p is None:
            self.closeup()
            self.forced = savedF
            return
        v, i = p
        fs = self.fs
        d = self.dg[v]
        k = fs[v][i]
        g = fs[v][(i - 1) % d]
        nv = len(fs)
        moves = []
        for u in range(nv):
            if u == v or u in self.nb[v]:
                continue
            fu = fs[u]
            du = self.dg[u]
            for j in range(du):
                if self.nb[u][j] == -1 and fu[(j - 1) % du] == k and fu[j] == g:
                    moves.append((u, j, None))
        if nv < self.maxV and self.dsum <= self.TOT:
            for fsw in self.newcands.get((k, g), []):
                key = self.ckey[fsw]
                if self.tidx[key] < self.curmin:
                    continue
                if self.need is not None and self.tcount[key] >= self.need.get(key, 0):
                    continue
                if self.dsum + self.dfx[fsw] > self.TOT:
                    continue
                moves.append((None, 0, fsw))

        if depth < len(self.prefix):
            idx = self.prefix[depth]
            moves = moves[idx:idx + 1] if idx < len(moves) else []
        elif self.split_depth is not None and depth == self.split_depth:
            for idx in range(len(moves)):
                self.tasks.append(tuple(self.path[:depth]) + (idx,))
            self.forced = savedF
            return

        for idx, (u, j, fsw) in enumerate(moves):
            realidx = self.prefix[depth] if depth < len(self.prefix) else idx
            self.path[depth] = realidx
            m = self.mark()
            if fsw is None:
                ok = self.link(v, i, u, j)
            else:
                w = self.add_vertex(fsw)
                ok = self.link(v, i, w, 0)
            if ok:
                self.dfs(depth + 1)
            self.undo(m)
            if self.hitcap:
                self.forced = savedF
                return
        self.forced = savedF

    # ----------------------------------------------------------------- closure
    def closeup(self):
        if self.dsum > self.TOT:
            return
        if self.need is not None:
            for key, c in self.need.items():
                if self.tcount[key] != c:
                    return
        fs, nb = self.fs, self.nb
        nv = len(fs)
        for v in range(nv):
            if -1 in nb[v]:
                return
        seen = set()
        faces = []
        for v in range(nv):
            for i in range(len(fs[v])):
                if (v, i) in seen:
                    continue
                chain = []
                x, c = v, i
                while True:
                    chain.append((x, c))
                    if len(chain) > 200:
                        return
                    y = nb[x][c]
                    jy = nb[y].index(x)
                    x, c = y, (jy - 1) % len(fs[y])
                    if (x, c) == (v, i):
                        break
                if len(chain) != fs[v][i]:
                    return
                vs = [a for a, _ in chain]
                if len(set(vs)) != len(vs):
                    return
                seen.update(chain)
                faces.append(vs)
        E = sum(len(nb[v]) for v in range(nv)) // 2
        if nv - E + len(faces) != 2:
            return
        # Dedup: the exact canonical planar code is O(E^2) and dominated the
        # whole run, so screen with a cheap WL hash first and only pay for the
        # exact code when two maps actually collide.
        from known import canon_code, wl
        h = wl(faces, nv)
        bucket = self.found.setdefault(h, [])
        if bucket:
            code = canon_code(faces, nv)
            if code is None:
                return
            for e in bucket:
                if e[0] is None:
                    e[0] = canon_code(e[1], e[2])
                if e[0] == code:
                    return
        else:
            code = None
        rec = (faces, nv, dict(self.tcount))
        bucket.append([code, faces, nv, rec])
        self.nfound += 1
        if self.verbose:
            fc = Counter(len(f) for f in faces)
            print(f"  FOUND #{self.nfound}: V={nv} F={len(faces)} "
                  f"faces={dict(sorted(fc.items()))} "
                  f"types={ {'.'.join(map(str,k2)): c for k2, c in sorted(self.tcount.items()) if c} }",
                  flush=True)
        if self.onfind:
            self.onfind(rec)


if __name__ == '__main__':
    types = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [[3, 3, 3, 3, 4], [3, 4, 4, 4]]
    name = sys.argv[2] if len(sys.argv) > 2 else 'tmap_out.json'
    maxnodes = float(sys.argv[3]) if len(sys.argv) > 3 else float('inf')
    out = []

    def save(rec):
        faces, nv, tc = rec
        out.append((faces, nv, {'.'.join(map(str, k)): c for k, c in tc.items() if c}))
        json.dump(out, open(name, 'w'))

    G = Gen(types, verbose=True, maxnodes=maxnodes, onfind=save)
    print(f"vocabulary {types}  maxV={G.maxV}", flush=True)
    t0 = time.time()
    G.run()
    print(f"done: {G.nfound} maps, {G.nodes} nodes, {time.time()-t0:.1f}s, "
          f"EXHAUSTIVE={not G.hitcap}")
    json.dump(out, open(name, 'w'))
