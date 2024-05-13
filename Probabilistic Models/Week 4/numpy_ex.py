import numpy as np

a = [1, 2]
b = [3, 4]
c = np.array([a, b])

print(np.dot(a, b))  # 1*3 + 2*4 = 11
print(np.matmul(a, b))  # 1*3 + 2*4 = 11
print(np.multiply(a, b))  # [1*3, 2*4] = [3, 8]
print(c.T)  # Transpose of c
