import cvxpy as cp
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt


#%matplotlib qt


n = 100
u = np.linspace(-1,1,n)

A = np.array([[u], [-u], [u**2], [-u**2], [3 * ( u**3 ) - 2 * u], [-3 * ( u**3 ) + 2 * u], [((u < 0)*1).tolist()]]).squeeze()
b = np.array([0.1, 0.1, 0.5, -0.5, -0.2, 0.3, 0.4])


p = cp.Variable(n)
constr = []
constr += [A@p <= b]
constr += [cp.sum(p) ==1]
constr += [p >= 0]

cost = cp.sum(cp.entr(p))
prob = cp.Problem(cp.Maximize(cost), constr)

prob.solve()

p = p.value

plt.plot(u, p)
plt.xlim(-1.3, 1.3)
plt.ylim(0, 0.05)
plt.xlabel('$u_i$', fontsize = 16)
plt.ylabel('$ Prob( X == u_i ) $', fontsize = 16)
#plt.rc('legend',fontsize=20) # using a size in points
#plt.rc('legend',fontsize='medium') # using a named size
ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.grid()
plt.legend()
#
plt.title('Maximum Entropy Distribution', fontsize = 16, color = 'k')
plt.tight_layout()
plt.show()
