"""sanity check of release OFF files: edge ratio, tangency to the unit sphere, barycentre"""
import sys, glob, os, numpy as np
bad = 0
files = sorted(glob.glob(os.path.join(sys.argv[1], "off", "*", "*.off")))
for fn in files[:: max(1, len(files) // 300)]:
    L = [l for l in open(fn) if l.strip() and not l.startswith("#")]
    nv, nf, ne = map(int, L[1].split())
    V = np.array([[float(x) for x in l.split()] for l in L[2:2 + nv]])
    F = [list(map(int, l.split()[1:])) for l in L[2 + nv:2 + nv + nf]]
    E = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in F for i in range(len(f))}
    A = V[[a for a, b in E]]; B = V[[b for a, b in E]]
    D = B - A
    s = -(A * D).sum(1) / (D * D).sum(1)
    T = A + s[:, None] * D
    dist = np.linalg.norm(T, axis=1)
    L_ = np.linalg.norm(D, axis=1)
    hdr = open(fn).read().split("edge ratio ")[1].split()[0]
    ok = np.allclose(dist, 1, atol=1e-6) and np.allclose(T.mean(0) / np.linalg.norm(T, axis=1).mean(), 0, atol=1e-6) \
        and abs(L_.max() / L_.min() - float(hdr)) < 1e-5 and len(E) == ne
    bad += not ok
    if not ok:
        print("BAD", fn, dist.min(), dist.max(), T.mean(0), L_.max() / L_.min(), hdr)
print("checked", len(files[:: max(1, len(files) // 300)]), "of", len(files), "bad", bad)
