"""Legal vertex-type table (the '127 table') generated from first principles,
plus tstar (tangent length of a regular-faced vertex of that type, midsphere radius 1).

Rules (user spec):
  - convex: angle sum < 360 strictly (flat 360 and hyperbolic >360 banned)
  - degree >= 3
  - banned infinite families: 3.3.x, 3.4.x, 3.5.x, 3.6.x, 4.4.x (degree-3)
                              3.3.3.x (degree-4)
  - cyclic order matters (up to rotation + reflection)
"""
import math
from itertools import product
from functools import lru_cache

KMAX = 60  # face-size cap; the angle condition cuts well below this


def ang(k):
    return 180.0 * (k - 2) / k


def canon(seq):
    """canonical representative of a cyclic sequence up to rotation and reflection"""
    n = len(seq)
    best = None
    for s in (list(seq), list(seq)[::-1]):
        for i in range(n):
            r = tuple(s[i:] + s[:i])
            if best is None or r < best:
                best = r
    return best


def banned(ty):
    """the six deliberately excluded infinite families"""
    d = len(ty)
    if d == 3:
        s = tuple(sorted(ty))
        if s[0] == 3 and s[1] in (3, 4, 5, 6):
            return True          # 3.3.x  3.4.x  3.5.x  3.6.x
        if s[0] == 4 and s[1] == 4:
            return True          # 4.4.x
        return False
    if d == 4:
        # 3.3.3.x : three consecutive triangles in cyclic order
        for i in range(4):
            if ty[i] == 3 and ty[(i + 1) % 4] == 3 and ty[(i + 2) % 4] == 3:
                return True
        return False
    return False


def gen_types():
    """recursive build with angle-sum pruning (each further face adds >= 60 deg)"""
    out = set()

    def rec(pre, s, d):
        if len(pre) == d:
            if s < 360 - 1e-9 and not banned(tuple(pre)):
                out.add(canon(tuple(pre)))
            return
        rem = d - len(pre) - 1          # faces still to add after this one
        for k in range(3, KMAX + 1):
            s2 = s + ang(k)
            if s2 + 60.0 * rem >= 360 - 1e-9:
                break                    # ang(k) increasing in k -> no larger k works
            rec(pre + [k], s2, d)

    for d in (3, 4, 5):
        rec([], 0.0, d)
    return sorted(out, key=lambda t: (len(t), t))


def tstar(ty):
    """tangent length of the regular-faced vertex cone, midsphere radius 1.
    Cone half-angle a solves sum_f arccos((cos th_f - cos^2 a)/sin^2 a) = 2pi ; t = 1/tan(a)...
    (verified below against known solids)"""
    from scipy.optimize import brentq
    th = [math.radians(ang(k)) for k in ty]

    def g(a):
        s = 0.0
        for t in th:
            c = (math.cos(t) - math.cos(a) ** 2) / math.sin(a) ** 2
            s += math.acos(max(-1.0, min(1.0, c)))
        return s - 2 * math.pi
    a = brentq(g, 1e-6, math.pi / 2 - 1e-12)
    return 1 / math.tan(a)


if __name__ == '__main__':
    T = gen_types()
    print("legal vertex types:", len(T))
    from collections import Counter
    print("by degree:", Counter(len(t) for t in T))
    # sanity: known solids
    for ty, name in [((4, 4, 4), 'cube'), ((3, 3, 3), 'tetra'), ((3, 3, 3, 3, 3), 'icosa'),
                     ((3, 4, 3, 4), 'cubocta'), ((3, 10, 10), 'trunc dodeca'),
                     ((4, 6, 10), 'trunc icosidodeca'), ((3, 8, 8), 'trunc cube')]:
        print(f"  tstar{ty} = {tstar(ty):.6f}   ({name})")
    print()
    rows = [(tstar(t), t) for t in T]
    rows.sort()
    print("tstar range over the table: %.4f .. %.4f" % (rows[0][0], rows[-1][0]))
    print("\n-- 25 smallest tstar --")
    for v, t in rows[:25]:
        print(f"  {v:9.5f}  {'.'.join(map(str,t))}")
    print("\n-- 25 largest tstar --")
    for v, t in rows[-25:]:
        print(f"  {v:9.5f}  {'.'.join(map(str,t))}")
