import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp


#%matplotlib qt




p = np.random.multivariate_normal([0,0], [[10,0],[0,10]],  10)


x = cp.Variable((1,p.shape[-1]))

constr = []
cost = cp.sum(cp.norm(x - p ,1))
prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = False)



xp = x.value


# -------  Display
plt.plot(p[:,0], p[:,1], 'k*')
plt.plot(xp[:,0], xp[:,1], 'ro')
plt.title('One free point localization')
plt.show()