import numpy as np
import matplotlib.pyplot as plt
import scipy

import cvxpy as cp


%matplotlib qt



x0 = np.array([5, -5])
m = 2
x = cp.Variable(m)

constr = []
constr += [x >= 0]

prob = cp.Problem(cp.Minimize(cp.norm(x - x0)), constr) 
prob.solve(verbose = False)



xp = x.value



#-------- Display


plt.plot(xp[0], xp[1], 'k*')
plt.plot(x0[0], x0[1], 'ro')
plt.xlim(-10,10)
plt.ylim(-10,10)
#plt.axis('off')