import numpy as np
p = np.random.permutation(10)

print(p)
a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([11,12,13,14,15,16,17,18,19,20])

x = a[p]
z = b[p]
print(x)
print(z)