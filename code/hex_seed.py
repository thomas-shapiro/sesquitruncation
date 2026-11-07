'''Sesquitruncation paper, audit step: the old run (sesqgen/sesqpar) left out the
parent vertex type 3.3.3 (child big face n=6, child vertex 3.6.n) and filtered
children through the 127-type table.  This run covers exactly the missing part:
parents containing at least one 3.3.3 vertex (tmap3 seed pass 0 = minimum type
3.3.3), dodecagon vertices allowed, child legality = strictly spherical
(sum of regular angles < 360), NOT the 127 table.
maxV=44 is a proven cap: non-dodecagon vertices each spend >=1 of the budget 24
(so N<=24) and have degree <=5; dodecagon vertices are pairwise non-adjacent and
have degree 6, so 6|D| <= 5N <= 120, |D| <= 20, V <= 44.
'''
import sys, os, json, time
from fractions import Fraction
sys.path.insert(0, os.path.expanduser('~/nmj'))
import tmap3
from tmap3 import canon_cyclic
from sesq import sesq
from mapcheck import orient, validate
from tprofile import cyclic_types

TYPES = {(3,3,3):6, (3,3,4):7, (3,4,4):8, (4,4,4):9, (3,3,3,3):8, (3,3,3,4):9,
         (3,3,4,4):10, (3,4,3,4):10, (3,4,4,4):11, (3,3,3,3,3):10,
         (3,3,3,3,4):11, (3,3,3,3,3,3):12}
TYPES = {canon_cyclic(t): n for t, n in TYPES.items()}
D12 = canon_cyclic((3,3,3,3,3,3))

def spherical(t):
    return sum(Fraction(180) - Fraction(360, k) for k in t) < 360

out, bad = [], 0
def onfind(rec):
    global bad
    F, nv, tc = rec
    child, cnv, info = sesq([list(f) for f in F], nv)
    if child is None: return
    child = [list(f) for f in orient(child)]
    if validate(child, cnv): return
    ts = cyclic_types(child, cnv).values()
    if not all(spherical(t) for t in ts):
        bad += 1; return
    out.append({'parent': [list(f) for f in F], 'pnv': nv, 'ptypes': {str(k): v for k, v in dict(tc).items()},
                'child': child, 'cnv': cnv, 'big': info['bigsizes']})

G = tmap3.Gen([list(t) for t in TYPES], weights={t: 12-n for t, n in TYPES.items()},
              total=24, maxV=44, onfind=onfind, verbose=True)
G.forbid = {(D12, D12)}
assert G.types0[0] == canon_cyclic((3,3,3)), G.types0[0]
G.seedfilter = 0
t0 = time.time(); G.run()
print(f'parents with a 3.3.3 vertex: {G.nfound}  legal children: {len(out)}  non-spherical dropped: {bad}  '
      f'nodes={G.nodes}  {time.time()-t0:.0f}s  EXHAUSTIVE={not G.hitcap}', flush=True)
json.dump(out, open(os.path.expanduser('~/nmj/sesqpaper/out6/hex_children.json'), 'w'))
