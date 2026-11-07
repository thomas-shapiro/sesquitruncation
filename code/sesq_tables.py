r"""LaTeX tables and the ratio histogram for the paper, from ext2/stats.json.
    python3 sesq_tables.py  -> latex/sections/auto/*.tex
"""
import json, os
from collections import Counter
from itertools import combinations
ALL = json.load(open(os.path.expanduser('~/nmj/sesqpaper/ext2/stats.json')))
FAM = {x['id']: x for x in json.load(open(os.path.expanduser('~/nmj/sesqpaper/ext2/family_ids.json')))}


def polyhedral_parent(F, n):
    """P is a polyhedron iff its dual P* (the parent) is a simple 3-connected planar map"""
    adj = [set() for _ in range(n)]
    for f in F:
        for j in range(len(f)):
            a, b = f[j], f[(j + 1) % len(f)]
            adj[a].add(b); adj[b].add(a)
    if 2 * sum(map(len, adj)) != 2 * sum(len(f) for f in F) or min(map(len, adj)) < 3:
        return False
    for x, y in combinations(range(n), 2):
        rest = [v for v in range(n) if v not in (x, y)]
        seen, st = {rest[0]}, [rest[0]]
        while st:
            u = st.pop()
            for v in adj[u] - seen - {x, y}:
                seen.add(v); st.append(v)
        if len(seen) != len(rest):
            return False
    return True


for r in ALL:
    r['poly'] = polyhedral_parent(FAM[r['id']]['parent'], FAM[r['id']]['pnv'])
TRIV = [r for r in ALL if r['poly'] and r['V4'] == 0]   # trivial: P simple, sigma(P) = truncation
S = [r for r in ALL if r['poly'] and r['V4'] > 0]       # proper sesquitruncates
DEG = [r for r in ALL if not r['poly']]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'latex', 'sections', 'auto')
os.makedirs(OUT, exist_ok=True)
EOL = ' \\\\'


def w(name, s):
    open(os.path.join(OUT, name), 'w').write(s + '\n')


def big_str(b):
    return ', '.join('$%d\\times%d$' % (v, int(k)) for k, v in sorted(b.items(), key=lambda kv: int(kv[0])))


# largest big face x number of dodecagons
mf = lambda r: max(int(k) for k in r['big'])
dd = lambda r: r['big'].get('12', 0)
rows = sorted(Counter(mf(r) for r in S))
cols = sorted(Counter(dd(r) for r in S))
C = Counter((mf(r), dd(r)) for r in S)
L = ['\\begin{tabular}{r|' + 'r' * len(cols) + '|r}', '\\toprule',
     'largest & \\multicolumn{%d}{c|}{number of dodecagons} & total' % len(cols) + EOL,
     'face & ' + ' & '.join(map(str, cols)) + ' & ' + EOL, '\\midrule']
for m in rows:
    L.append('%d & ' % m + ' & '.join(str(C[(m, d)]) if C[(m, d)] else '' for d in cols)
             + ' & %d' % sum(C[(m, d)] for d in cols) + EOL)
L += ['\\midrule', 'total & ' + ' & '.join(str(sum(C[(m, d)] for m in rows)) for d in cols)
      + ' & %d' % len(S) + EOL, '\\bottomrule', '\\end{tabular}']
w('tab_faces.tex', '\n'.join(L))

# by V (bins of 20)
B = Counter((r['V'] // 20) * 20 for r in S)
ks = sorted(B)
L = ['\\begin{tabular}{l' + 'r' * len(ks) + '}', '\\toprule',
     '$V(\\st P)$ & ' + ' & '.join(('%d--%d' % (k, k + 19)) if k else '$\\le19$' for k in ks) + EOL, '\\midrule',
     'members & ' + ' & '.join(str(B[k]) for k in ks) + EOL, '\\bottomrule', '\\end{tabular}']
w('tab_V.tex', '\n'.join(L))

# by symmetry group, sorted by order
G = Counter(r['group'] for r in S)
order = {r['group']: r['order'] for r in S}
gs = sorted(G, key=lambda g: (-order[g], g))
L = ['\\begin{tabular}{lrr@{\\qquad}lrr@{\\qquad}lrr}', '\\toprule',
     'group & order & members & group & order & members & group & order & members' + EOL, '\\midrule']
third = (len(gs) + 2) // 3
for i in range(third):
    cells = []
    for j in range(3):
        k = i + j * third
        cells.append('$%s$ & %d & %d' % (gs[k], order[gs[k]], G[gs[k]]) if k < len(gs) else ' & & ')
    L.append(' & '.join(cells) + EOL)
L += ['\\bottomrule', '\\end{tabular}']
w('tab_groups.tex', '\n'.join(L))

# members with rho < 1.1
NAMES = {'S0001': 'triangular hosohedron$^{*}$', 'S0002': 'tetrahedron', 'S0013': 'cube',
         'S1101': 'dodecahedron', 'S7525': 'deltoidal icositetrahedron', 'S7524': 'dual of $J_{37}$',
         'S7513': 'pentagonal icositetrahedron', 'S7520': 'dual of $J_{45}$', 'S2385': 'dual of $J_{27}$',
         'S2386': 'rhombic dodecahedron', 'S0368': 'octahedron', 'S2128': 'pentagonal bifrustum',
         'S7527': 'order-6 trunc.\\ tetrakis hexahedron'}
best = sorted((r for r in S if r['canon'] < 1.1), key=lambda r: (round(r['canon'], 6), r['id']))
L = ['\\begin{tabular}{llrrrllr}', '\\toprule',
     'member & $P$ & $F(P)$ & $V$ & $F$ & big faces & group & $\\ratio$' + EOL, '\\midrule']
for r in best:
    L.append('%s & %s & %d & %d & %d & %s & $%s$ & %s' % (
        r['id'], NAMES.get(r['id'], ''), r['inF'], r['V'], r['F'], big_str(r['big']), r['group'],
        '$1$' if r['canon'] < 1 + 1e-9 else '%.4f' % r['canon']) + EOL)
L += ['\\bottomrule', '\\end{tabular}']
w('tab_best.tex', '\n'.join(L))

# histogram of rho (bins of 0.05 from 1 to 2, last bin rho >= 2)
H = Counter(min(int((r['canon'] - 1) * 20 + 1e-9), 20) for r in S)
Y = 1.0 / 600   # cm per member
L = ['\\begin{tikzpicture}[xscale=0.6]']
for b in range(21):
    h = H[b]
    L.append('\\filldraw[fill=gray!35,draw=black,line width=0.3pt] (%d,0) rectangle (%d,%.3f);' % (b, b + 1, h * Y))
    L.append('\\node[above,font=\\tiny] at (%.1f,%.3f) {%d};' % (b + 0.5, h * Y, h))
L.append('\\draw[->] (0,0) -- (21.8,0) node[right,font=\\scriptsize] {$\\ratio$};')
for b in range(0, 21, 2):
    L.append('\\node[below,font=\\scriptsize] at (%d,0) {%.1f};' % (b, 1 + b * 0.05))
L.append('\\node[below,font=\\scriptsize] at (20.5,-0.25) {$\\ge2$};')
L.append('\\end{tikzpicture}')
w('fig_hist.tex', '\n'.join(L))

chiral = sum(1 for r in S if r['group'] in ('C_{1}', 'C_{2}', 'C_{3}', 'C_{4}', 'T', 'O', 'I')
             or (r['group'].startswith('D_{') and r['group'][-2] not in 'hd'))
w('degenerate.tex', '\n'.join('%s & %d & %d & %s & $%s$ & %.4f%s' % (r['id'], r['inF'], r['V'], big_str(r['big']), r['group'],
                                r['canon'], EOL) for r in sorted(DEG, key=lambda r: r['canon'])[:8]))
json.dump([r['id'] for r in DEG], open(os.path.expanduser('~/nmj/sesqpaper/ext2/degenerate_ids.json'), 'w'))
print(json.dumps({'deg': len(DEG), 'deg_digon': sum(1 for r in DEG if min(len(FAM[r['id']]['parent'][0]), 9) and
                  any(sum(1 for f in FAM[r['id']]['parent'] if v in f) == 2 for v in range(FAM[r['id']]['pnv']))),
                  'deg_best': [(r['id'], r['V'], round(r['canon'], 4)) for r in sorted(DEG, key=lambda r: r['canon'])[:8]],
                  'deg_trunc': sum(r['V4'] == 0 for r in DEG), 'deg_four': sum(r['V3'] == 0 for r in DEG)}))
print(json.dumps({'N': len(S), 'trunc': sum(r['V4'] == 0 for r in S), 'four': sum(r['V3'] == 0 for r in S),
                  'below11': len(best), 'Vmin': min(r['V'] for r in S), 'Vmax': max(r['V'] for r in S),
                  'chiral': chiral, 'C1': G['C_{1}'], 'bins': {b: H[b] for b in range(21)}}))

import statistics
mx = lambda r: max(int(k) for k in r['big'])
print(json.dumps({
    'proper': len(S), 'trivial': len(TRIV),
    'no_face_gt10': sum(mx(r) <= 10 for r in S), 'has14': sum(mx(r) >= 14 for r in S),
    'max14_per': max(sum(v for k, v in r['big'].items() if int(k) >= 14) for r in S),
    'faces14plus_sizes': sorted({int(k) for r in S for k in r['big'] if int(k) >= 14}),
    'median': statistics.median(r['canon'] for r in S), 'ge2': sum(r['canon'] >= 2 for r in S),
    'b110_115': sum(1.1 <= r['canon'] < 1.15 for r in S),
    'best_groups': sorted({r['group'] for r in best}),
    'Vmin_member': min((r['V'], r['id']) for r in S), 'Vmax_member': max((r['V'], r['id']) for r in S),
    'triv_Vrange': [min(r['V'] for r in TRIV), max(r['V'] for r in TRIV)],
    'triv_below11': sorted((r['id'], round(r['canon'], 4)) for r in TRIV if r['canon'] < 1.1),
    'triv_best_nonreg': sorted((round(r['canon'], 4), r['id'], r['V']) for r in TRIV if r['canon'] > 1 + 1e-9)[:3],
    'triv_maxface': sorted(Counter(mx(r) for r in TRIV).items()),
    'triv_14plus': sum(mx(r) >= 14 for r in TRIV),
}))
