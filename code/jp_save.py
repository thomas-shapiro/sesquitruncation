import sys, numpy as np
import jp_fit as J
x = np.array([float(a) for a in sys.argv[1:6]])
print("ratio", J.score(x, verbose=True))
P, lab = J.make(x)
np.save("sesq4d_1.npy", P)
