import numpy as np

x = np.array([1, -1]).reshape(1, 2)
W_x = np.array([[1 / 2, 1 / 2], [0, 1], [1, 0]])
W_h = np.array([-1, 1 / 2, 1 / 2]).reshape(3, 1)

"""

Back propagation can always find the global optimum regardless of weights initialization.
0.5
0.269
0.731
0.5
0.125
0.625
"""


def sigmoid(var):
    return 1 / (1 + np.exp(-var))


def d_sigmoid(var):
    return var * (1 - var)


def d_loss(y, a_v):
    return -y / a_v


z1 = W_x @ x.T
print(z1)
h = sigmoid(z1)
print(h)
print("--" * 3, "a")
z = h.T @ W_h
a = sigmoid(z)
print(a)

"""
Backpropagation
"""

d_s = -1/a
print("Ds")
print(d_s)
d_z = d_s * d_sigmoid(a)
print("Dz")
print(d_z)
print("d_h")
dh = d_z * W_h
print("d_ z1")
print(z1)
print("d_z1")
d_z1 = dh * d_sigmoid(h)
print(d_z1)
dw = d_z1 * x
print("dw")
print(dw)

print("wq")
wq = W_x - 0.5 * dw
print(wq)
