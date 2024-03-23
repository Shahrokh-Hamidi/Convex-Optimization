import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp

#%matplotlib qt


n = 4

    
A = cp.Variable((n,n), PSD = True)


constr = []

constr += [A[0,0] == 3]
constr += [A[1,1] == 2]
constr += [A[2,2] == 1]
constr += [A[3,3] == 5]
constr += [A[0,1] == 0.5]
constr += [A[0,3] == 0.25]
constr += [A[1,2] == 0.75]

cost = -cp.log_det(A)
prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = True)


print(f'A = {A.value}')
