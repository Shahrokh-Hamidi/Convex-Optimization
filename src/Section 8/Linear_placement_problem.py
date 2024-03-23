import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp


#%matplotlib qt

fixed = np.array([[ 1,   1,  -1, -1,    1,   -1,  -0.2,  0.1], [ 1,  -1,  -1,  1, -0.5, -0.2,    -1,    1]]).T
M = fixed.shape[0]
N = 6

# first N columns of A correspond to free points,
# last M columns correspond to fixed points

A = np.array([[ 1,  0,  0, -1,  0,  0,    0,  0,  0,  0,  0, 0,  0, 0],
              [1,  0, -1,  0,  0,  0,    0,  0,  0,  0,  0,  0,  0,  0],
      [1,  0,  0,  0, -1,  0,    0,  0,  0,  0,  0,  0,  0,  0],
      [1,  0,  0, 0,  0,  0,   -1,  0,  0,  0,  0,  0,  0,  0],
      [1,  0,  0,  0,  0,  0,    0, -1,  0,  0,  0,  0,  0,  0],
      [1,  0, 0,  0,  0, 0,    0,  0,  0,  0, -1, 0,  0,  0],
      [1,  0,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0,  0, -1],
      [0,  1, -1,  0,  0,  0,    0,  0,  0,  0,  0,  0,  0,  0],
      [0,  1,  0, -1,  0,  0,    0,  0,  0,  0,  0,  0,  0,  0],
      [0,  1,  0,  0,  0, -1,    0,  0,  0, 0,  0,  0,  0,  0],
      [0,  1,  0,  0, 0,  0,    0, -1,  0,  0,  0,  0,  0,  0],
      [0,  1,  0,  0,  0,  0,    0,  0, -1,  0,  0,  0,  0,  0],
      [0,  1,  0,  0,  0,  0,    0,  0,  0,  0,  0,  0, -1, 0],
      [0,  0,  1, -1,  0,  0,    0,  0,  0,  0,  0,  0,  0,  0],
      [0,  0,  1,  0,  0,  0,    0, -1,  0,  0,  0,  0,  0,  0],
      [0,  0,  1,  0,  0,  0,    0,  0,  0,  0, -1,  0,  0,  0],
      [0,  0,  0,  1, -1,  0,    0,  0,  0,  0,  0,  0,  0,  0],
      [0,  0,  0,  1,  0,  0,    0,  0, -1,  0,  0,  0,  0,  0],
      [0,  0,  0,  1,  0,  0,    0,  0,  0, -1,  0,  0,  0,  0],
      [0,  0,  0,  1,  0,  0,    0,  0,  0,  0,  0, -1,  0,  0],
      [0,  0,  0,  1,  0, -1,    0,  0,  0,  0,  0, -1,  0,  0] ,  
      [0,  0,  0,  0,  1, -1,    0,  0,  0,  0,  0,  0,  0,  0],
      [0,  0,  0,  0,  1,  0,   -1,  0,  0,  0,  0,  0,  0,  0],
      [0,  0,  0,  0,  1,  0,    0,  0,  0, -1,  0,  0,  0,  0],
      [0,  0,  0,  0,  1,  0,    0,  0,  0,  0,  0,  0,  0, -1],
      [0,  0,  0,  0,  0,  1,    0,  0, -1,  0,  0,  0,  0,  0],
      [0,  0,  0,  0,  0,  1,    0,  0,  0,  0, -1,  0,  0,  0]])


x = cp.Variable((A.shape[-1], 2))
cost = cp.sum_squares(A@x)

constr = []
for i in range(A.shape[0]):
    constr += [x[N:,:] == fixed]

prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = False)



x = x.value

#------ Display

plt.plot(x[:N,0], x[:N,1], 'gs', label = 'free points')
plt.plot(x[N:,0], x[N:,1], 'rs', label = 'fixed points')

for i in range(A.shape[0]):
    ind = np.nonzero(A[i,:])
    for idx in ind:
        plt.plot(x[idx,0],x[idx,1], 'k-.', lw = 0.2)
    
plt.legend()
plt.title('Linear Placement Problem')
plt.xlim(-1.2,1.2)
plt.ylim(-1.2,1.2)
plt.show()