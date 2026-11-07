"""face-angle deviations of the equilateral cells: sesq octahedron (cell of the sesq 24-cell) and
sesq rhombic dodecahedron (cell of the equilateral honeycomb)"""
import numpy as np
import sesq4d as S, sesq_honey as H


def angle_dev(P, F):
    worst = {}
    for f in F:
        n = len(f)
        reg = np.pi * (n - 2) / n
        for i in range(n):
            a, b, c = P[f[i - 1]], P[f[i]], P[f[(i + 1) % n]]
            u, v = a - b, c - b
            ang = np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
            d = abs(ang - reg) / reg
            worst[n] = max(worst.get(n, 0), d)
    return {n: round(100 * d, 2) for n, d in sorted(worst.items())}


phi = (1 + 5 ** .5) / 2
s = 2 ** .5 / (1 + 2 ** .5 * phi ** 2); t4 = phi ** 2 * s / 2
V, E, F, C, cub = S.show("24-cell", S.cell24())
P, lab = S.sesq_points(V, E, F, 0.3, s, cub, t4)
E2, F2, C2 = S.faces4(P)
big = max(C2, key=len)
idx = sorted(big)
sub = [f for f in F2 if set(f) <= big]
# express the cell in 3D coordinates
X = P[idx] - P[idx].mean(0)
B = np.linalg.svd(X)[2][:3]
Q = np.zeros((len(P), 3)); Q[idx] = X @ B.T
print("sesq octahedron cell: faces", sorted({len(f) for f in sub}), "max angle deviation % by face size", angle_dev(Q, sub))
t = (33 - 6 * 6 ** .5) / 97
P3, F3 = H.sesq_cell(np.zeros(3), t, 2 * t, 8 * t / 3)
print("sesq rhombic dodecahedron cell:", angle_dev(P3, F3))
