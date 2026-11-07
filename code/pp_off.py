import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsesq import gsesq, pyramid, orient, canonical_robust
F = orient(pyramid(5)); G, n = gsesq(F); G = orient(G)
V, d = canonical_robust(G, n)
with open("degen/sesq_pentpyr.off", "w") as f:
    f.write("OFF\n%d %d 0\n" % (len(V), len(G)))
    for p in V: f.write("%.12f %.12f %.12f\n" % tuple(p))
    for g in G: f.write("%d %s\n" % (len(g), " ".join(map(str, g))))
print(n, d["ratio"])
