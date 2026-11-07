import numpy as np, os, sys, glob, math, json
from collections import Counter, defaultdict

def read_off(path):
    lines=[l.split('#')[0] for l in open(path, errors='ignore').read().splitlines()]
    lines=[l for l in lines if l.strip()]
    assert lines[0].strip().startswith('OFF')
    hdr=lines[1].split(); nv,nf=int(hdr[0]),int(hdr[1])
    V=np.array([[float(x) for x in l.split()[:3]] for l in lines[2:2+nv]])
    F=[]
    for l in lines[2+nv:2+nv+nf]:
        t=l.split(); k=int(t[0]); F.append([int(x) for x in t[1:1+k]])
    return V,F
    i = 1
    nv, nf, ne = int(toks[i]), int(toks[i+1]), int(toks[i+2]); i += 3
    V = np.array([float(t) for t in toks[i:i+3*nv]]).reshape(nv,3); i += 3*nv
    F = []
    for _ in range(nf):
        k = int(toks[i]); F.append([int(t) for t in toks[i+1:i+1+k]]); i += k+1
    return V, F

def analyze(path):
    V, F = read_off(path)
    nv, nf = len(V), len(F)
    # edges
    E = set()
    for f in F:
        for a in range(len(f)):
            u, v = f[a], f[(a+1)%len(f)]
            E.add((min(u,v), max(u,v)))
    E = sorted(E)
    ne = len(E)
    # centroid of vertices -> shift? canonical should be centered already. Check midsphere: edge tangent distances
    C = V.mean(axis=0)
    def edge_dist(u, v, c):
        p, q = V[u]-c, V[v]-c
        d = q-p
        t = -np.dot(p, d)/np.dot(d, d)
        t = min(max(t,0),1)
        return np.linalg.norm(p + t*d)
    eds = np.array([edge_dist(u,v,np.zeros(3)) for u,v in E])
    midsphere_spread = (eds.max()-eds.min())/eds.mean()
    eds_c = np.array([edge_dist(u,v,C) for u,v in E])
    midsphere_spread_c = (eds_c.max()-eds_c.min())/eds_c.mean()
    lens = np.array([np.linalg.norm(V[u]-V[v]) for u,v in E])
    ratio = lens.max()/lens.min()
    # tangent lengths
    r = eds.mean()
    t = np.sqrt(np.maximum(np.linalg.norm(V,axis=1)**2 - r**2, 0))
    # vertex figures: faces around each vertex in cyclic order
    vf = defaultdict(list)
    for fi, f in enumerate(F):
        for a in range(len(f)):
            vf[f[a]].append(fi)
    # cyclic order around vertex: use face adjacency via edges
    def cyc(v):
        fs = vf[v]
        # for each face, the two neighbors of v in it
        nb = {}
        for fi in fs:
            f = F[fi]; k = len(f); a = f.index(v)
            nb[fi] = (f[(a-1)%k], f[(a+1)%k])
        order = [fs[0]]; cur = fs[0]; nxt_v = nb[cur][1]
        while len(order) < len(fs):
            for fi in fs:
                if fi not in order and nxt_v in nb[fi]:
                    order.append(fi); cur = fi
                    nxt_v = nb[fi][1] if nb[fi][0]==nxt_v else nb[fi][0]
                    break
            else:
                break
        return [len(F[fi]) for fi in order]
    def canon_cyc(seq):
        # canonical form up to rotation and reflection
        n = len(seq); best = None
        for s in (seq, seq[::-1]):
            for i in range(n):
                r_ = tuple(s[i:]+s[:i])
                if best is None or r_ < best: best = r_
        return best
    types = []
    defects = []
    face_angles_at = []
    for v in range(nv):
        seq = cyc(v)
        types.append(canon_cyc(seq))
        # actual angle defect
        s = 0
        for fi in vf[v]:
            f = F[fi]; k = len(f); a = f.index(v)
            p, q = V[f[(a-1)%k]]-V[v], V[f[(a+1)%k]]-V[v]
            ang = math.degrees(math.acos(np.dot(p,q)/np.linalg.norm(p)/np.linalg.norm(q)))
            s += ang
        defects.append(360-s)
    defects = np.array(defects)
    # ideal defects from regular faces
    ideal = np.array([360 - sum(180*(k-2)/k for k in ty) for ty in types])
    # interior angles deviation from regular
    max_dev = 0
    for f in F:
        k = len(f); reg = 180*(k-2)/k
        for a in range(k):
            p, q = V[f[(a-1)%k]]-V[f[a]], V[f[(a+1)%k]]-V[f[a]]
            ang = math.degrees(math.acos(np.dot(p,q)/np.linalg.norm(p)/np.linalg.norm(q)))
            max_dev = max(max_dev, abs(ang-reg))
    facesizes = Counter(len(f) for f in F)
    degs = Counter(len(vf[v]) for v in range(nv))
    tc = Counter(types)
    return dict(name=os.path.basename(path), V=nv, F=nf, E=ne, euler=nv-ne+nf,
                ratio=ratio, mid_spread=midsphere_spread, mid_spread_c=midsphere_spread_c,
                r=r, t_min=t.min(), t_max=t.max(), t_ratio=t.max()/t.min(),
                def_min=defects.min(), def_max=defects.max(), span=defects.max()-defects.min(),
                ideal_def_min=ideal.min(), ideal_def_max=ideal.max(), ideal_span=ideal.max()-ideal.min(),
                max_angle_dev=max_dev, faces=dict(sorted(facesizes.items())), degs=dict(sorted(degs.items())),
                types={'.'.join(map(str,k)): c for k,c in sorted(tc.items(), key=lambda kv:-kv[1])},
                emin=lens.min(), emax=lens.max())

if __name__ == '__main__':
    out = []
    for d in sys.argv[1:]:
        for p in sorted(glob.glob(d+'/*.off')):
            try:
                out.append(analyze(p))
            except Exception as e:
                out.append(dict(name=os.path.basename(p), error=str(e)))
    json.dump(out, open('/home/claude/results_%s.json' % os.path.basename(sys.argv[1]), 'w'), indent=1, default=float)
    for o in out:
        if 'error' in o: print(o); continue
        print(f"{o['name'][:60]:60s} V{o['V']:4d} F{o['F']:4d} chi{o['euler']} ratio {o['ratio']:.5f} tR {o['t_ratio']:.4f} midsp {o['mid_spread']:.1e} span {o['span']:.2f} (ideal {o['ideal_span']:.2f}) def[{o['def_min']:.2f},{o['def_max']:.2f}] angdev {o['max_angle_dev']:.2f} faces {o['faces']} degs {o['degs']}")
