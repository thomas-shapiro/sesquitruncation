"""Hart-style canonicalization (edges tangent to the unit sphere, tangency barycentre at 0, planar faces)."""
import sys, numpy as np
from ana_degen_io import read_off
def canon(V, F, it=20000, tol=1e-12):
    E = np.array(sorted({tuple(sorted((f[i], f[(i+1)%len(f)]))) for f in F for i in range(len(f))}))
    V = V - V.mean(0)
    for k in range(it):
        A, B = V[E[:,0]], V[E[:,1]]; D = B-A
        s = -(A*D).sum(1)/(D*D).sum(1); T = A+s[:,None]*D
        r = np.linalg.norm(T, axis=1)
        mv = np.zeros_like(V); cnt = np.zeros(len(V))
        corr = T*((1-r)/r)[:,None]
        np.add.at(mv, E[:,0], corr); np.add.at(mv, E[:,1], corr)
        np.add.at(cnt, E[:,0], 1); np.add.at(cnt, E[:,1], 1)
        V = V + 0.5*mv/cnt[:,None]
        V = V - T.mean(0)
        # planarize
        mv = np.zeros_like(V); cnt = np.zeros(len(V))
        for f in F:
            P = V[f]; c = P.mean(0); n = np.linalg.svd(P-c)[2][2]
            d = ((P-c)@n)
            np.add.at(mv, f, -0.5*d[:,None]*n[None,:]); np.add.at(cnt, f, 1)
        V = V + mv/cnt[:,None]
        if k % 200 == 0:
            pl = max(np.linalg.svd(V[f]-V[f].mean(0))[1][-1] for f in F)
            err = max(np.abs(r-1).max(), np.linalg.norm(T.mean(0)), pl)
            if err < tol: break
    L = np.linalg.norm(V[E[:,0]]-V[E[:,1]], axis=1)
    return V, err, L.max()/L.min()
for fn in sys.argv[1:]:
    V, F = read_off(fn)
    W, err, rho = canon(V, F)
    print(fn, "err %.1e ratio %.5f" % (err, rho))
    np.save(fn.replace(".off", "_canon.npy"), W)
