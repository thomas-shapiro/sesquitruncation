"""Full sesquitruncation family (paper version, generator 1).

Parent = dual P* of the {3,4}-polyhedron P: a map with triangles and squares.
Parent vertex types = words in {3,4}; type -> big face n = 2t + 3s.
Allowed (Corollary 3.8): n <= 11 (11 types), n = 12 (3^6), n = 2k for 7 <= k <= 11 (3^k).
Edge rule (Prop 3.7b): 1/n + 1/n' > 1/6 on every parent edge (forbid pairs).
Budget (Lemma 5.4): every N-vertex (n <= 11) has final charge >= LB(n) and the final
charges of N sum to exactly 24, so  sum LB(n) <= 24  is a valid prune.
LB = 1, 10/7, 2, 24/11, 45/11, 56/11 for n = 11, 10, 9, 8, 7, 6 (scaled by 77).
maxV = 32 (Theorem 5.5 iii).
Usage: python3 ext_gen.py <seed index | all> [maxnodes]
"""
import sys, os, json, time
from fractions import Fraction
sys.path.insert(0, os.path.expanduser('~/nmj'))
sys.path.insert(0, os.path.expanduser('~/nmj/sesqpaper'))
import tmap3x as tmap3          # copy of tmap3 with an "at most TOTAL" budget
from tmap3x import canon_cyclic
from sesq import sesq
from mapcheck import orient, validate
from tprofile import cyclic_types

WORDS = {(3,3,3):6, (3,3,4):7, (3,4,4):8, (4,4,4):9, (3,3,3,3):8, (3,3,3,4):9,
         (3,3,4,4):10, (3,4,3,4):10, (3,4,4,4):11, (3,3,3,3,3):10, (3,3,3,3,4):11,
         (3,)*6: 12, (3,3): 4, (3,4): 5, (4,4): 6}
for k in range(7, 12):
    WORDS[(3,)*k] = 2*k
TYPES = {canon_cyclic(w): n for w, n in WORDS.items()}
LB77 = {11: 77, 10: 110, 9: 154, 8: 168, 7: 315, 6: 392}
WEIGHT = {t: LB77.get(n, 0) for t, n in TYPES.items()}
WEIGHT.update({canon_cyclic((3,3)): 8*77, canon_cyclic((3,4)): 7*77, canon_cyclic((4,4)): 6*77})
TOTAL = 24 * 77
FORBID = {(a, b) for a, na in TYPES.items() for b, nb in TYPES.items()
          if Fraction(1, na) + Fraction(1, nb) <= Fraction(1, 6)}


def spherical(t):
    return sum(Fraction(180) - Fraction(360, k) for k in t) < 360


def make(seed=None, maxnodes=float('inf'), onfind=None):
    G = tmap3.Gen([list(t) for t in TYPES], weights=WEIGHT, total=TOTAL, maxV=32,
                  maxnodes=maxnodes, onfind=onfind, verbose=False)
    G.forbid = set(FORBID)
    if seed is not None:
        G.seedfilter = seed
    return G


if __name__ == '__main__':
    seed = None if sys.argv[1] == 'all' else int(sys.argv[1])
    maxnodes = float(sys.argv[2]) if len(sys.argv) > 2 else float('inf')
    outdir = os.path.expanduser('~/nmj/sesqpaper/ext2')
    os.makedirs(outdir, exist_ok=True)
    out, stats = [], {'nonspherical': 0, 'invalid': 0}

    def onfind(rec):
        F, nv, tc = rec
        child, cnv, info = sesq([list(f) for f in F], nv)
        if child is None:
            stats['invalid'] += 1; return
        child = [list(f) for f in orient(child)]
        if validate(child, cnv):
            stats['invalid'] += 1; return
        if not all(spherical(t) for t in cyclic_types(child, cnv).values()):
            stats['nonspherical'] += 1; return
        out.append({'parent': [list(f) for f in F], 'pnv': nv, 'child': child, 'cnv': cnv,
                     'big': info['bigsizes']})

    G = make(seed, maxnodes, onfind)
    t0 = time.time()
    G.run()
    tag = 'all' if seed is None else f'seed{seed:02d}'
    print(f"{tag} type={G.types0[seed] if seed is not None else '-'} parents={G.nfound} "
          f"children={len(out)} {stats} nodes={G.nodes} {time.time()-t0:.0f}s "
          f"EXHAUSTIVE={not G.hitcap}", flush=True)
    json.dump({'seed': seed, 'exhaustive': not G.hitcap, 'nodes': G.nodes, 'stats': stats,
               'children': out}, open(f'{outdir}/{tag}.json', 'w'))
