"""Robust canonicalisation: retry the midsphere solve from perturbed starts.

`develop.canonical` runs one Newton/hybr solve from a single deterministic
initial guess and gives up if it does not converge.  Every 3-connected planar
map HAS a midsphere realisation, so a failure there is numerical, not real -
and an unmeasured closure is a hole in an exhaustive claim.  This retries with
scaled and randomised starts before declaring failure.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser('~/nmj'))
from mapcheck import orient, validate
import develop as D
from radii import RadiiSolver


def canonical_robust(F, nv, tries=40, seed=0):
    F = orient(F)
    if validate(F, nv):
        return None
    S = RadiiSolver(F, nv)
    x00 = S.initial()
    rng = np.random.default_rng(seed)
    best = None
    for k in range(tries):
        if k == 0:
            x0 = x00
        elif k <= 6:
            x0 = x00 * (0.6 + 0.15 * k)          # uniform rescaling
        else:
            x0 = np.abs(x00 * (1.0 + rng.normal(0, 0.35, x00.shape))) + 1e-6
        try:
            t, r, res, info = S.solve(x0=x0)
        except Exception:
            continue
        if res <= 1e-9 and np.all(np.isfinite(t)):
            best = (t, r, res)
            break
    if best is None:
        return None
    t, r, res = best
    n, p, V0 = D.develop(F, nv, t, r)
    V, P, cres = D.normalise(F, nv, p)
    d = D.check(V, F)
    d['solver_resid'] = res
    return V, d


if __name__ == '__main__':
    import json, glob
    from collections import Counter
    from legal import legal_types, canon_cyclic
    from tprofile import cyclic_types
    from known import tag
    LEGAL = set(legal_types(range(3, 42)))
    files = sorted(glob.glob(os.path.expanduser('~/nmj/t2/*/_solverfail/*.json')))
    print('records: %d' % len(files), flush=True)
    stat = Counter()
    hits = []
    for p in files:
        d = json.load(open(p))
        F = [list(f) for f in d['faces']]
        nv = d['nv']
        try:
            Fo = [list(f) for f in orient(F)]
        except Exception:
            stat['orient-failed'] += 1
            continue
        if validate(Fo, nv):
            stat['not-a-valid-map'] += 1
            continue
        ty = [canon_cyclic(t) for t in cyclic_types(Fo, nv).values()]
        if any(t not in LEGAL for t in ty):
            stat['illegal-vertex-type'] += 1
            continue
        r = canonical_robust(Fo, nv)
        if r is None:
            stat['STILL-UNMEASURED'] += 1
            hits.append(('unmeasured', nv, len(Fo), os.path.basename(p)))
            continue
        V, dd = r
        if dd['ratio'] < 1.1 and not dd['faces_with_vertices_outside']:
            stat['UNDER-1.1'] += 1
            hits.append(('ratio=%.5f' % dd['ratio'], nv, len(Fo), tag(Fo, nv)))
            print('*** ratio %.5f V=%d F=%d archive=%s'
                  % (dd['ratio'], nv, len(Fo), tag(Fo, nv)), flush=True)
        else:
            stat['resolved-fail'] += 1
    print(dict(stat), flush=True)
    for h in hits[:30]:
        print('   ', h)
