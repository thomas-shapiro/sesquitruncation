import numpy as np
import sesq_honey as H, sesq4d as S
t = (33 - 6 * np.sqrt(6)) / 97
print("honeycomb ratio at exact t:", H.ratio((t, 2 * t, 8 * t / 3)))
phi = (1 + 5 ** .5) / 2
s = 2 ** .5 / (1 + 2 ** .5 * phi ** 2); t4 = phi ** 2 * s / 2
V, E, F, C, cub = S.show("24-cell", S.cell24())
P, lab = S.sesq_points(V, E, F, 0.3, s, cub, t4)
E2, F2, C2, cells, r = S.census4(P)
print("24-cell ratio at exact s,t:", r, "edge", np.linalg.norm(P[E2[0][0]] - P[E2[0][1]]), "pred", s)
