"""How close is the canonical tangent length t_v to tstar(type(v)) in real near-misses?

If t_v ~ tstar(type) tightly, then 'ratio < 1.1' becomes an almost-purely COMBINATORIAL
condition on which vertex types may coexist / be adjacent -> massive pruning.
"""
import sys, glob, os, math, json
import numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, os.path.expanduser('~/nmj'))
from analyze import read_off
from mapcheck import orient, validate
from radii import RadiiSolver, edges_of
from develop import develop, normalise
from vtypes import tstar, gen_types, canon


def cyclic_types(F, nv):
    """vertex -> cyclic tuple of incident face sizes (canonical up to rot+refl)"""
    vf = defaultdict(list)
    for fi, f in enumerate(F):
        for v in f:
            vf[v].append(fi)
    out = {}
    for v in range(nv):
        fs = vf[v]
        nb = {}
        for fi in fs:
            f = F[fi]; k = len(f); a = f.index(v)
            nb[fi] = (f[(a - 1) % k], f[(a + 1) % k])
        order = [fs[0]]; cur = fs[0]; nxt = nb[cur][1]
        while len(order) < len(fs):
            for fi in fs:
                if fi not in order and nxt in nb[fi]:
                    order.append(fi)
                    nxt = nb[fi][1] if nb[fi][0] == nxt else nb[fi][0]
                    break
            else:
                break
        out[v] = canon(tuple(len(F[fi]) for fi in order))
    return out


def solve(path):
    V, F = read_off(path)
    nv = len(V)
    F = orient(F)
    if validate(F, nv):
        return None
    S = RadiiSolver(F, nv)
    t, r, res, info = S.solve()
    if res > 1e-9 or not np.all(np.isfinite(t)):
        return None
    n, p, V0 = develop(F, nv, t, r)
    Vc, P, cres = normalise(F, nv, p)
    tt = np.sqrt(np.maximum(np.linalg.norm(Vc, axis=1) ** 2 - 1.0, 0))
    E = edges_of(F)
    L = np.array([tt[u] + tt[v] for u, v in E])
    return dict(F=F, nv=nv, t=tt, E=E, ratio=L.max() / L.min(), types=cyclic_types(F, nv))


if __name__ == '__main__':
    TS = {ty: tstar(ty) for ty in gen_types()}
    rows = []
    alldev = []
    typeobs = defaultdict(list)
    for d in sys.argv[1:]:
        for path in sorted(glob.glob(d + '/*.off')):
            try:
                r = solve(path)
            except Exception as e:
                print(os.path.basename(path), 'ERR', e); continue
            if r is None:
                print(os.path.basename(path), 'solver failed'); continue
            devs = []
            illegal = []
            for v in range(r['nv']):
                ty = r['types'][v]
                if ty not in TS:
                    illegal.append(ty); continue
                devs.append(r['t'][v] / TS[ty])
                typeobs[ty].append(r['t'][v])
            devs = np.array(devs)
            alldev.extend(devs.tolist())
            tsv = [TS[r['types'][v]] for v in range(r['nv']) if r['types'][v] in TS]
            spread = (max(tsv) / min(tsv)) if tsv else float('nan')
            rows.append((os.path.basename(path), r['ratio'], devs.min() if len(devs) else 0,
                         devs.max() if len(devs) else 0, spread, len(illegal)))
            print(f"{os.path.basename(path)[:38]:38s} ratio {r['ratio']:.5f} "
                  f"t/tstar in [{devs.min():.4f},{devs.max():.4f}] "
                  f"tstar-spread {spread:.4f}  illegal-types {len(illegal)}")
    a = np.array(alldev)
    print()
    print(f"ALL {len(a)} vertices:  t/tstar  min {a.min():.5f}  max {a.max():.5f}  "
          f"mean {a.mean():.5f}  std {a.std():.5f}")
    print()
    print("observed t per type (over all hits):")
    for ty in sorted(typeobs, key=lambda x: -len(typeobs[x])):
        o = np.array(typeobs[ty])
        print(f"  {'.'.join(map(str,ty)):18s} n={len(o):5d}  tstar {TS[ty]:.5f}  "
              f"t obs [{o.min():.5f},{o.max():.5f}]  ratio-to-tstar [{o.min()/TS[ty]:.4f},{o.max()/TS[ty]:.4f}]")
