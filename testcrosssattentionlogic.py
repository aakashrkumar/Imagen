import numpy as np

b = 1
h = 10
w = 10
d = 15

# x = np.ones((b, h, w, d))
# s = np.ones((b, ))

x = np.ones((1, 15))
s = np.ones((21, 15))

q = x
v = s + x
k = s + x

y = np.dot(v.transpose(), k)
y = np.dot(q, y)
print(y.shape)