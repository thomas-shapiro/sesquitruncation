import numpy as np, math
from collections import defaultdict, deque
from radii import *

def rot_about(axis, ang):
    axis = axis/np.linalg.norm(axis); K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + math.sin(ang)*K + (1-math.cos(ang))*K@K

def develop(F, nv, t, r):
    """Place face-circle centres n_f on the unit sphere and edge tangency points p_e from (t, r).
    Returns n (F x 3), p (dict edge->3), and vertex coordinates V."""
    nf = len(F); rho = np.sin(r)
    # edge -> (face, position) incidences and face adjacency in cyclic order
    edge_faces = defaultdict(list)
    for fi, f in enumerate(F):
        for a in range(len(f)):
            u, v = f[a], f[(a+1) % len(f)]; edge_faces[(min(u,v),max(u,v))].append((fi, a))
    def other(e, fi): return [g for g, _ in edge_faces[e] if g != fi][0]
    theta = {}   # (fi, v) -> face angle at v
    for fi, f in enumerate(F):
        for v in f: theta[(fi, v)] = 2*math.atan2(rho[fi], t[v])
    n = np.zeros((nf, 3)); placed = np.zeros(nf, bool)
    p = {}
    # seed: face 0 at north pole, first edge tangency point toward +x
    n[0] = [0, 0, 1]; placed[0] = True
    q = deque([0])
    # for each placed face we need one known tangency direction; store it
    ref = {0: (0, np.array([1.0, 0, 0]))}   # face -> (edge index a, unit tangent direction at n_f)
    def unit_dir(nf_, d):  # make d tangent at nf_ and unit
        d = d - np.dot(d, nf_)*nf_; return d/np.linalg.norm(d)
    while q:
        fi = q.popleft(); f = F[fi]; k = len(f)
        a0, d0 = ref[fi]
        # walk around the face: edge a runs from f[a] to f[a+1]; the centre angle between successive
        # tangency points (edges a-1 and a) equals pi - theta at vertex f[a]
        d = unit_dir(n[fi], d0)
        dirs = [None]*k; dirs[a0] = d
        for step in range(1, k):
            a = (a0 + step) % k
            v = f[a]                       # vertex between edge a-1 and edge a
            d = rot_about(n[fi], -(math.pi - theta[(fi, v)])) @ d   # orientation sign fixed below by consistency
            dirs[a] = unit_dir(n[fi], d)
        for a in range(k):
            u, v = f[a], f[(a+1) % k]; e = (min(u,v), max(u,v))
            pe = math.cos(r[fi])*n[fi] + math.sin(r[fi])*dirs[a]
            if e in p:
                pass
            else:
                p[e] = pe
            g = other(e, fi)
            if not placed[g]:
                ang = r[fi] + r[g]
                n[g] = math.cos(ang)*n[fi] + math.sin(ang)*dirs[a]
                placed[g] = True
                # in g, the same edge appears; its tangency direction from n_g points back toward n_f
                ag = [b for (h, b) in edge_faces[e] if h == g][0]
                ref[g] = (ag, unit_dir(n[g], n[fi] - n[g]))
                q.append(g)
    V = np.zeros((nv, 3)); ve = defaultdict(list)
    for e in p:
        ve[e[0]].append(p[e]); ve[e[1]].append(p[e])
    for v in range(nv):
        A = np.array(ve[v]); V[v] = np.linalg.lstsq(A, np.ones(len(A)), rcond=None)[0]
    return n, p, V

def boost(P, b):
    """Lorentz boost of unit vectors P (N x 3) by velocity vector b (|b|<1)."""
    beta = np.linalg.norm(b)
    if beta < 1e-300: return P
    u = b/beta; g = 1/math.sqrt(1-beta**2)
    pu = P@u
    num = P + ((g-1)*pu)[:,None]*u + g*beta*u
    den = g*(1 + beta*pu)
    return num/den[:,None]

def normalise(F, nv, p, iters=100):
    """Möbius-normalise so the tangency-point centroid is at the origin; return vertex coords, tangency pts."""
    keys = list(p); P = np.array([p[e] for e in keys])
    for it in range(iters):
        c = P.mean(0)
        if np.linalg.norm(c) < 1e-15: break
        P = boost(P, -c)          # push centroid toward origin
    ve = defaultdict(list)
    for e, pe in zip(keys, P): ve[e[0]].append(pe); ve[e[1]].append(pe)
    V = np.zeros((nv, 3))
    for v in range(nv):
        A = np.array(ve[v]); V[v] = np.linalg.lstsq(A, np.ones(len(A)), rcond=None)[0]
    return V, dict(zip(keys, P)), np.linalg.norm(P.mean(0))

def check(V, F):
    """convexity + planarity + midsphere diagnostics"""
    E = edges_of(F)
    def ld(u, v):
        a, b = V[u], V[v]; d = b-a; s = -np.dot(a, d)/np.dot(d, d); return np.linalg.norm(a+s*d)
    eds = np.array([ld(u, v) for u, v in E])
    planar = 0; nonconvex = 0
    for f in F:
        P = V[f]; c = P.mean(0); n = np.linalg.svd(P-c)[2][2]
        if np.dot(n, c) < 0: n = -n
        planar = max(planar, np.abs((P-c)@n).max())
        d = (V - P[0])@n
        if d.max() > 1e-8: nonconvex += 1
    L = np.array([np.linalg.norm(V[u]-V[v]) for u, v in E])
    return dict(midsphere_resid=np.ptp(eds), planarity=planar, faces_with_vertices_outside=nonconvex, ratio=L.max()/L.min())

def canonical(F, nv):
    from mapcheck import orient, validate
    F = orient(F)
    if validate(F, nv): return None
    S = RadiiSolver(F, nv); t, r, res, info = S.solve()
    if res > 1e-9 or not np.all(np.isfinite(t)): return None
    n, p, V0 = develop(F, nv, t, r)
    V, P, cres = normalise(F, nv, p)
    d = check(V, F); d['solver_resid'] = res; d['centroid_resid'] = cres
    return V, d

if __name__ == '__main__':
    import sys, glob, os
    for dd in sys.argv[1:]:
        for path in sorted(glob.glob(dd+'/*.off')):
            V, F = read_off(path)
            out = canonical(F, len(V))
            if out is None: print(os.path.basename(path)[:36], 'solver failed'); continue
            Vc, d = out
            print(f"{os.path.basename(path)[:36]:36s} ratio {d['ratio']:.5f} mid {d['midsphere_resid']:.1e} planar {d['planarity']:.1e} nonconvex {d['faces_with_vertices_outside']} centroid {d['centroid_resid']:.1e}")
