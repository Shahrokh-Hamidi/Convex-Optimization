import numpy as np
import matplotlib.pyplot as plt 
import cvxpy as cp

#%matplotlib qt

n=6
m=40


u = np.linspace(-1,1,m)
v = 1/(5+40*u**2) + 0.1*u**3 + np.random.normal(0,0.01,m)



A = np.vander(u)[:,-n:]
plt.plot(u,v, 'o', mfc = 'none')



def Poly_fitting(method):
    
    x = cp.Variable(n)
    
    cost = cp.norm(A@x - v, method) 
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    
    return x.value

method = 2
x_l2 = Poly_fitting(method)

method = 1
x_l1 = Poly_fitting(method)

method = 'inf'
x_inf = Poly_fitting(method)

plt.plot(u, v, 'bo', mfc = 'none', label = 'Data')
plt.plot(u, A@x_l1, 'k', label = 'l1 norm')
plt.plot(u, A@x_l2, 'r', label = 'l2 norm')
plt.plot(u, A@x_inf, 'g', label = '$l_\infty$ norm')
plt.xlabel('u', fontsize = 16)
plt.ylabel('v', fontsize = 16)
plt.rc('legend',fontsize=20) # using a size in points
#plt.rc('legend',fontsize='medium') # using a named size
ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
plt.grid()
plt.legend()
#
plt.title('Polynomial Fitting', fontsize = 16)
plt.tight_layout()
plt.show()
