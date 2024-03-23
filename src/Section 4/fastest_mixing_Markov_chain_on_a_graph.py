import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp

#%matplotlib qt


n = 5
E = np.array([[0, 1, 0, 1, 1],
              [1, 0, 1, 0, 1],
              [0, 1, 0, 1, 1],
              [1, 0, 1, 0, 1],
              [1, 1, 1, 1, 0]])


P = cp.Variable((n,n), symmetric = True)



constr = []
constr += [P@np.ones(n) == np.ones(n)]
constr += [P >= 0]
constr += [P[E==0] == 0] 

cost = cp.norm(P - np.ones((n,n))/n)
prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = True)


print(f'the optimal value is {np.round(P.value,2)}')
