"""Coordinate-free canonical solver.
Unknowns: x_v = log t_v (tangent length of vertex v), r_f (angular radius of face incircle, rho_f = sin r_f).
Equations: face angle sums and vertex cone closures. Newton via scipy.optimize.root with analytic sparse Jacobian.
"""
import numpy as np, math, time, sys, glob, os
from collections import defaultdict
from scipy.optimize import root
from scipy.sparse import coo_matrix
from analyze import *

def edges_of(F):
    return sorted({(min(f[a], f[(a+1) % len(f)]), max(f[a], f[(a+1) % len(f)])) for f in F for a in range(len(f))})

def tstar(ty):
    """natural tangent length of a vertex type with regular faces"""
    from scipy.optimize import brentq
    th = [math.radians(180*(k-2)/k) for k in ty]
    def g(a):
        return sum(math.acos(max(-1, min(1, (math.cos(t)-math.cos(a)**2)/math.sin(a)**2))) for t in th) - 2*math.pi
    return 1/math.tan(brentq(g, 1e-4, math.pi/2-1e-9))

class RadiiSolver:
    def __init__(self, F, nv):
        self.F = F; self.nv = nv; self.nf = len(F)
        self.inc_f = np.array([fi for fi, f in enumerate(F) for _ in f])
        self.inc_v = np.array([v for f in F for v in f])
        self.k = np.array([len(f) for f in F])
        self.deg = np.bincount(self.inc_v, minlength=nv)
        self.vtypes = defaultdict(list)
        for fi, v in zip(self.inc_f, self.inc_v): self.vtypes[v].append(self.k[fi])
        n = nv + self.nf
        # Jacobian sparsity: face rows depend on r_f and t_v (v in f); vertex rows on t_v and r_f (f at v)
        rows = np.concatenate([self.inc_f, self.inc_f, self.nf + self.inc_v, self.nf + self.inc_v])
        cols = np.concatenate([self.inc_v, nv + self.inc_f, self.inc_v, nv + self.inc_f])
        self.rows, self.cols = rows, cols

    def initial(self):
        t0 = np.array([tstar(tuple(self.vtypes[v])) for v in range(self.nv)])
        tbar = t0.mean()
        rho0 = np.clip(tbar / np.tan(np.pi / self.k), 1e-3, 0.999)
        return np.concatenate([np.log(t0), np.arcsin(rho0)])

    def fun_jac(self, x):
        nv, nf = self.nv, self.nf
        t = np.exp(x[:nv]); r = x[nv:]; rho = np.sin(r); cr = np.cos(r)
        tv = t[self.inc_v]; rf = rho[self.inc_f]; crf = cr[self.inc_f]
        D = tv**2 + rf**2
        theta = 2*np.arctan2(rf, tv)
        # face residuals
        Ff = np.bincount(self.inc_f, weights=theta, minlength=nf) - (self.k - 2)*np.pi
        dth_dt = -2*rf/D                       # d theta / d t_v
        dth_dr = 2*tv*crf/D                    # d theta / d r_f
        # vertex residuals: phi = arccos(c), c = (t^2 - rho^2 - 2 t^2 rho^2)/(t^2+rho^2)
        N = tv**2 - rf**2 - 2*tv**2*rf**2
        c = N/D
        c = np.clip(c, -1+1e-15, 1-1e-15)
        phi = np.arccos(c)
        Fv = np.bincount(self.inc_v, weights=phi, minlength=nv) - 2*np.pi
        dN_dt = 2*tv - 4*tv*rf**2; dN_dr = (-2*rf - 4*tv**2*rf)*crf
        dD_dt = 2*tv; dD_dr = 2*rf*crf
        dc_dt = (dN_dt*D - N*dD_dt)/D**2; dc_dr = (dN_dr*D - N*dD_dr)/D**2
        s = -1/np.sqrt(1 - c**2)
        dphi_dt = s*dc_dt; dphi_dr = s*dc_dr
        # chain rule for x = log t: d/dx = t d/dt
        vals = np.concatenate([dth_dt*tv, dth_dr, dphi_dt*tv, dphi_dr])
        J = coo_matrix((vals, (self.rows, self.cols)), shape=(nf+nv, nv+nf)).toarray()
        return np.concatenate([Ff, Fv]), J

    def solve(self, x0=None, tol=1e-13):
        if x0 is None: x0 = self.initial()
        res = root(lambda x: self.fun_jac(x)[0], x0, jac=lambda x: self.fun_jac(x)[1], method='hybr', tol=tol)
        x = res.x; f, _ = self.fun_jac(x)
        t = np.exp(x[:self.nv]); r = x[self.nv:]
        r = np.arcsin(np.abs(np.sin(r)))   # equations depend on sin r only; fold to (0, pi/2)
        return t, r, np.abs(f).max(), res

def edge_ratio(t, E):
    L = np.array([t[u] + t[v] for u, v in E]); return L.max()/L.min()

def from_coords(V, F):
    """extract (t, r) from a canonical coordinate file for validation"""
    E = edges_of(F)
    def ld(u, v):
        p, q = V[u], V[v]; d = q-p; s = -np.dot(p, d)/np.dot(d, d); return np.linalg.norm(p+s*d)
    R = np.mean([ld(u, v) for u, v in E]); V = V/R
    t = np.sqrt(np.linalg.norm(V, axis=1)**2 - 1)
    r = []
    for f in F:
        P = V[f]; c = P.mean(0); n = np.linalg.svd(P-c)[2][2]
        if np.dot(n, c) < 0: n = -n
        r.append(math.acos(min(1, abs(np.dot(n, P[0])))))
    return t, np.array(r)

if __name__ == '__main__':
    for d in sys.argv[1:]:
        for p in sorted(glob.glob(d + '/*.off')):
            V, F = read_off(p); E = edges_of(F)
            S = RadiiSolver(F, len(V))
            t_true, r_true = from_coords(V, F)
            f_true, _ = S.fun_jac(np.concatenate([np.log(t_true), r_true]))
            t0 = time.time(); t, r, res, info = S.solve(); dt = time.time()-t0
            true_ratio = edge_ratio(t_true, E)
            print(f"{os.path.basename(p)[:36]:36s} V{len(V):3d} | eqs at file coords: {np.abs(f_true).max():.1e} | solver: residual {res:.1e} in {dt*1000:6.1f} ms, {info.nfev:4d} evals | ratio solver {edge_ratio(t,E):.5f} vs file {true_ratio:.5f} | max|t-t_true|/t {np.abs(t-t_true).max()/t_true.mean():.1e}")
