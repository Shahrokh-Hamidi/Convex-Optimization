import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp


#%matplotlib qt

n = 2
A = np.eye(n)
b = np.zeros(n)
C = np.array([[2, 1], [-.5, 1]])
d = np.array([-3, -3])


x = cp.Variable(n)
y = cp.Variable(n)
w = cp.Variable(n)

constr = []

constr += [cp.norm(A@x + b, 2) <= 1]
constr += [cp.norm(C@y + d, 2) <= 1]
constr += [x - y == w]
cost = cp.norm(w, 2)
prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = False)

w = w.value
x = x.value
y = y.value
# -------  Display


angle = np.linspace(0,2*np.pi,100).reshape(1,-1)
points = np.vstack((np.cos(angle), np.sin(angle)))

ellipse_1 = np.linalg.inv(A)@(points - b.reshape(-1,1))
ellipse_2 = np.linalg.inv(C)@(points - d.reshape(-1,1))

z = (x+y)/2.
m = (y[1] - x[1])/(y[0] - x[0])
m_ = (-90 + np.arctan(m)*180/np.pi)*np.pi/180
b_ = z[1] - m_*z[0]

t = np.linspace(-1.5, 2, 100)
plt.plot(t, m_*t + b_, 'm')
plt.plot([x[0], y[0]], [x[1], y[1]])
plt.plot(x[0], x[1], 'kx', markersize = 10)
plt.plot(y[0], y[1], 'rx', markersize = 10)
plt.plot(ellipse_1[0,:], ellipse_1[1,:], 'b--')
plt.plot(ellipse_2[0,:], ellipse_2[1,:], 'g--')
plt.xlim(-2,4)
plt.ylim(-2,4)
plt.title('Separating Ellipsoids in 2D')
plt.show()