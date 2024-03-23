import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp


#%matplotlib qt

n = 2
px = np.array([-5, 5])
py = np.array([-10, 10])



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

x0 = [-5, 5]
m = 2
x = cp.Variable(m)

constr = []
constr += [A@x <= b]

prob = cp.Problem(cp.Minimize(cp.norm(x - x0, 2)), constr) 
prob.solve(verbose = False)



xp = x.value



#-------- Display

plt.plot(px, py, 'b', lw = 2)
plt.plot(xp[0], xp[1], 'k*', markersize = 10)
plt.plot(x0[0], x0[1], 'ro')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('Euclidean projection on a halfspace')
#plt.axis('off')
plt.show()