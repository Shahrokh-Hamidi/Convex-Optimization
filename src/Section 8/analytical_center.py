import numpy as np
import matplotlib.pyplot as plt
import scipy

import cvxpy as cp


#%matplotlib qt

n = 2
px = np.array([0, .5, 2, 3, 1])
py = np.array([0, 1, 1.5, .5, -.5])

px = np.hstack((px, px[0]))
py = np.hstack((py, py[0]))

px_diff = px[1:] - px[:-1]
py_diff = py[1:] - py[:-1]

px_avg = 0.5*(px[1:] + px[:-1])
py_avg = 0.5*(py[1:] + py[:-1])


A = []
for i in range(0, len(px)-1):
    p = np.array([px_diff[i], py_diff[i]])
    p = p/np.linalg.norm(p)
    A.append([-p[1], p[0]])

A = np.array(A)

b = []
for i in range(0, len(px)-1):
    p = np.array([px_avg[i], py_avg[i]])
    b.append(A[i,:].dot(p))

#plt.plot(px, py)

m = A.shape[-1]
x = cp.Variable(m)

constr = []
constr += [A@x <= b]

prob = cp.Problem(cp.Minimize(-cp.sum(cp.log(b - A@x))), constr) 
prob.solve(verbose = False)



xp = x.value



#-------- Display

plt.plot(px, py, 'k')
plt.plot(xp[0], xp[1], 'ro')
plt.title('Analytical Center of a Polytope')
plt.axis('off')
plt.show()