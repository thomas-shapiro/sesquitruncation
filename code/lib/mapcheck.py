import networkx as nx
from collections import Counter, defaultdict, deque

def orient(F):
    """Return faces re-oriented consistently (BFS), so each undirected edge is traversed once in each direction."""
    F = [list(f) for f in F]
    ef = defaultdict(list)
    for fi, f in enumerate(F):
        for a in range(len(f)):
            u, v = f[a], f[(a+1) % len(f)]; ef[frozenset((u, v))].append(fi)
    done = [False]*len(F); done[0] = True; q = deque([0])
    while q:
        fi = q.popleft(); f = F[fi]
        for a in range(len(f)):
            u, v = f[a], f[(a+1) % len(f)]
            for g in ef[frozenset((u, v))]:
                if g == fi or done[g]: continue
                fg = F[g]; k = len(fg)
                # g must traverse v->u
                if any(fg[b] == u and fg[(b+1) % k] == v for b in range(k)): F[g] = fg[::-1]
                done[g] = True; q.append(g)
    return F

def validate(F, nv):
    """Combinatorial checks for a polyhedral map: returns list of problems (empty = OK)."""
    problems = []
    c = Counter()
    for f in F:
        if len(f) < 3: problems.append('face with <3 vertices')
        if len(set(f)) != len(f): problems.append('repeated vertex in a face')
        for a in range(len(f)): c[frozenset((f[a], f[(a+1) % len(f)]))] += 1
    if any(n != 2 for n in c.values()): problems.append('edge not in exactly two faces')
    E = len(c); Fn = len(F)
    if nv - E + Fn != 2: problems.append(f'Euler characteristic {nv-E+Fn} != 2')
    G = nx.Graph(); G.add_edges_from(tuple(e) for e in c)
    if G.number_of_nodes() != nv: problems.append('isolated vertex')
    elif min(d for _, d in G.degree()) < 3: problems.append('vertex of degree < 3')
    elif not nx.is_connected(G) or nx.node_connectivity(G) < 3: problems.append('not 3-connected (Steinitz fails)')
    # two faces sharing more than one edge (already implied by 3-connectivity for simple maps, but cheap)
    pair = Counter()
    ef = defaultdict(list)
    for fi, f in enumerate(F):
        for a in range(len(f)): ef[frozenset((f[a], f[(a+1) % len(f)]))].append(fi)
    for e, fs in ef.items():
        if len(fs) == 2: pair[tuple(sorted(fs))] += 1
    if any(n > 1 for n in pair.values()): problems.append('two faces share >1 edge')
    return problems
