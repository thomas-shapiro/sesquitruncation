import numpy as np
def read_off(fn):
    L = [l.split('#')[0].split() for l in open(fn)]
    L = [l for l in L if l]
    assert L[0][0].upper().endswith('OFF')
    nv, nf = int(L[1][0]), int(L[1][1])
    X = np.array([[float(x) for x in l[:3]] for l in L[2:2+nv]])
    F = [[int(x) for x in l[1:1+int(l[0])]] for l in L[2+nv:2+nv+nf]]
    return X, F
