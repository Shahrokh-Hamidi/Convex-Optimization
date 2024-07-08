import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib 



#
# the data has been generated based on an AR model of order 1
# when A0 = 1, the model describes random walk
# in order to get an AR model A0 should be chosen as A0 < 1




n = 2000
y = np.zeros(n)
A0 = 1
for i in range(1,n):
    y[i] = A0*y[i-1] + np.random.normal(0,0.05)

n = len(y)


A = np.hstack((np.array([1, -2, 1]), np.zeros(n-3)))

D = [] 
for i in range(len(A)):
    D.append(np.roll(A,i))

D = np.array(D)



lambda_ = 10
x = cp.Variable(n)

cost = cp.norm(y - x, 2) + lambda_* cp.sum(cp.abs(D@x))      #cp.tv(x)
constr = []
prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = True)



x = x.value

plt.rcParams['savefig.dpi'] = 300

plt.plot(y, 'k-', lw = 0.3)
plt.plot(x, 'r')
matplotlib.rc('font', size=14)
plt.title('Time Series Analysis')

plt.xlabel('t', fontsize = 16)
plt.ylabel('y[t]', fontsize = 16)
matplotlib.style.use('ggplot')
plt.show()

