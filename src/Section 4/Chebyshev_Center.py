import numpy as np
import matplotlib.pyplot as plt
import scipy

import cvxpy as cp


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
r = cp.Variable(1)

constr = []
constr += [A@x + r <= b]

prob = cp.Problem(cp.Maximize(r), constr) 
prob.solve(verbose = False)



x = x.value
r = r.value


#---- Display

N = 100

theta = np.linspace(0,2*np.pi, N).reshape(1,-1)

x1 = r*np.cos(theta).squeeze() + x[0]
x2 = r*np.sin(theta).squeeze() + x[1]


plt.plot(px, py)
plt.plot(x1, x2, 'r--')
plt.plot(x[0], x[1], 'ko')
plt.title('Chebyshev Center') 
plt.axis('off')
plt.show()

