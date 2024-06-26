import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp

#%matplotlib qt




lambda_ = 1

yns = np.array([[ 5.2057354 ],
       [ 5.16852954],
       [ 4.46931747],
       [ 3.16764149],
       [ 3.21867268],
       [ 2.75587742],
       [ 1.63606279],
       [ 0.72527756],
       [ 0.24583927],
       [-0.58044829],
       [-0.87676552],
       [-0.82548372],
       [-0.79731423],
       [-0.05948396],
       [-0.04975525],
       [ 0.70500264],
       [ 0.82600211],
       [ 0.1403059 ],
       [ 0.51054544],
       [ 0.38582234],
       [ 0.83860086],
       [ 0.41632982],
       [ 0.81154682],
       [ 0.23060127],
       [ 0.84177419],
       [ 0.34454159],
       [ 0.37408514],
       [ 0.86597229],
       [ 0.2120701 ],
       [ 0.71788999],
       [ 0.80995603],
       [ 1.06910934],
       [ 0.64850169],
       [ 1.09248439],
       [ 0.76143045],
       [ 1.2122857 ],
       [ 1.17728916],
       [ 0.84659501],
       [ 0.95866895],
       [ 1.82113178],
       [ 1.80159358],
       [ 1.63543887],
       [ 1.77429526],
       [ 2.52647668],
       [ 2.65227764],
       [ 3.75011515],
       [ 4.05642222],
       [ 4.62476811],
       [ 4.91230273],
       [ 5.80689459],
       [ 7.02609346]])

u = np.array([[0.  ],
       [0.04],
       [0.08],
       [0.12],
       [0.16],
       [0.2 ],
       [0.24],
       [0.28],
       [0.32],
       [0.36],
       [0.4 ],
       [0.44],
       [0.48],
       [0.52],
       [0.56],
       [0.6 ],
       [0.64],
       [0.68],
       [0.72],
       [0.76],
       [0.8 ],
       [0.84],
       [0.88],
       [0.92],
       [0.96],
       [1.  ],
       [1.04],
       [1.08],
       [1.12],
       [1.16],
       [1.2 ],
       [1.24],
       [1.28],
       [1.32],
       [1.36],
       [1.4 ],
       [1.44],
       [1.48],
       [1.52],
       [1.56],
       [1.6 ],
       [1.64],
       [1.68],
       [1.72],
       [1.76],
       [1.8 ],
       [1.84],
       [1.88],
       [1.92],
       [1.96],
       [2.  ]])


n = len(yns)
y = cp.Variable((n,1))
g = cp.Variable((n,1))


constr = []

constr += [y@np.ones((1,n)) >=  np.ones((n,1))@y.T + cp.multiply((np.ones((n,1))@g.T), (u@np.ones((1,n)) - np.ones((n,1))@u.T))]

cost = cp.norm(y - yns) 
prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = False)



y = y.value

plt.figure()
plt.plot(u, yns, 'o', mfc= 'none')
plt.plot(u, y, 'b')
plt.xlabel('u', fontsize = 16)
plt.ylabel('y', fontsize = 16)
ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
plt.title('Fitting a convex function to given data')
plt.xlim(-0.5,2.5)
plt.ylim(-1,8)
plt.grid()
plt.show()