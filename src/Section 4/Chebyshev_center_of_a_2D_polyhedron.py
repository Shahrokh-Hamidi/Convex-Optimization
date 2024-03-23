import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp

#%matplotlib qt



a1 = np.array([ 2, 1])
a2 = np.array([ 2, -1])
a3 = np.array([-1,  2])
a4 = np.array([-1, -2])
b = np.ones(4)


m = 2
c = cp.Variable(m)
r = cp.Variable(1)

constr = []

constr += [a1@c + np.linalg.norm(a1,2)*r <= b[0]]
constr += [a2@c + np.linalg.norm(a2,2)*r <= b[1]]
constr += [a3@c + np.linalg.norm(a3,2)*r <= b[2]]
constr += [a4@c + np.linalg.norm(a4,2)*r <= b[3]]

prob = cp.Problem(cp.Maximize(r), constr) 
prob.solve(verbose = False)



c = c.value
r = r.value


#---- Display

N = 100

theta = np.linspace(0,2*np.pi, N).reshape(1,-1)
x = np.linspace(-2,2,100)




x1 = r*np.cos(theta).squeeze() + c[0]
x2 = r*np.sin(theta).squeeze() + c[1]

plt.plot(x, b[0]/a1[1] - a1[0]*x/a1[1], 'k')
plt.plot(x, b[1]/a2[1] - a2[0]*x/a2[1], 'k')
plt.plot(x, b[2]/a3[1] - a3[0]*x/a3[1], 'k')
plt.plot(x, b[3]/a4[1] - a4[0]*x/a4[1], 'k')
plt.plot(x1, x2, 'r--')
plt.plot(c[0], c[1], 'bo')
plt.title('Chebyshev center of a 2D polyhedron') 
plt.ylim(-1,1)

plt.show()

