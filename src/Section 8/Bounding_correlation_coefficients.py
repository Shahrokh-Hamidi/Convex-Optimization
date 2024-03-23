import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp


#%matplotlib qt



m = 4
p = cp.Variable((m,m), PSD = True)
constr = []
constr += [p[0][1] <= 0.9]
constr += [0.6 <= p[0][1]]
constr += [p[0][2] <= 0.9]
constr += [0.8 <= p[0][2]]
constr += [p[1][3] <= 0.7]
constr += [0.5 <= p[1][3]]
constr += [p[2][3] <= -0.4]
constr += [-0.8 <= p[2][3]]
for i in range(m):
    constr += [p[i][i] == 1.]


prob = cp.Problem(cp.Maximize(p[0][3]), constr) 
prob.solve(verbose = False)



print(f'p.value = {p.value}')