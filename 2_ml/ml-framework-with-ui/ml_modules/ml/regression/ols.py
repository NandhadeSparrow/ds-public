n, p = [int(x) for x in input().split()]
X = []
for i in range(n):
    X.append([float(x) for x in input().split()])

y = [float(x) for x in input().split()]

import numpy as np
# X = np.array(X).reshape(n,p)
y=np.array(y)
b=np.linalg.pinv(X) @ y.T #inv
print(np.around(b,2))


import numpy as np

# Create the design matrix
X = np.array([[1, 2], [3, 4], [5, 6]])

# Create the response vector
y = np.array([7, 11, 15])

# Estimate the parameters
params, residuals = np.linalg.lstsq(X, y)

# Print the estimated parameters
print(params)