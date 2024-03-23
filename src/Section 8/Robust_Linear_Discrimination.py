import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp


#%matplotlib qt




x = np.random.multivariate_normal([-2,-2], [[1,0],[0,1]],  10)
y = np.random.multivariate_normal([2,2], [[1,0],[0,1]],  10)



a = cp.Variable((1,x.shape[-1]))
b = cp.Variable(1)
t = cp.Variable(1)

constr = []

for i in range(x.shape[0]):
    constr += [a@x[i,:].reshape(-1,1) - b >= t]
    constr += [a@y[i,:].reshape(-1,1) - b <= -t]
constr += [cp.norm(a,2) <= 1]
cost = t
prob = cp.Problem(cp.Maximize(cost), constr) 
prob.solve(verbose = False)

a = a.value.squeeze()
b = b.value
t = t.value
# -------  Display

z = np.linspace(-5,5,100)
plt.plot(z, (b - a[0]*z)/a[1])
plt.plot(z, (t + b - a[0]*z)/a[1], 'r--')
plt.plot(z, (-t + b - a[0]*z)/a[1], 'r--')
plt.plot(x[:,0], x[:,1], 'k*')
plt.plot(y[:,0], y[:,1], 'ro')
plt.title('Robust Linear Discrimination')
plt.show()
