import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp


#%matplotlib qt




x = np.random.multivariate_normal([-3,-3], [[1,0],[0,1]],  10)
y = np.random.multivariate_normal([3,3], [[1,0],[0,1]],  10)



a = cp.Variable((1,x.shape[-1]))
b = cp.Variable(1)


constr = []

for i in range(x.shape[0]):
    constr += [a@x[i,:].reshape(-1,1) - b >= 1]
    constr += [a@y[i,:].reshape(-1,1) - b <= -1]
cost = 0
prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = False)


# -------  Display

z = np.linspace(-5,5,100)
plt.plot(z, (b.value - a.value.squeeze()[0]*z)/a.value.squeeze()[1])
plt.plot(x[:,0], x[:,1], 'k*')
plt.plot(y[:,0], y[:,1], 'ro')
plt.title('Linear Discrimination')
plt.show()