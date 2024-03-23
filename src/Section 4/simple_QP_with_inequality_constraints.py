import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp




P = np.array([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])
q = np.array([[-22, -14.5, 13]])
r = 1
m = 3
#x_star = [1;1/2;-1];



x = cp.Variable(m)


constr = []

constr += [x <= 1]
constr += [x >= -1]

cost = 0.5*cp.quad_form(x, P) + q@x + r
prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = True)


x = x.value

print(f'x_optimal : {x}')
