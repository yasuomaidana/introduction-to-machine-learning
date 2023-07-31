import numpy as np


def g_from_list(Rs, gamma):
    gi = [np.power(gamma,i) * r for i, r in enumerate(Rs)]
    return sum(gi)


R = [-3, 5, 2, 7, 1]
print(g_from_list(R, 0.8))
