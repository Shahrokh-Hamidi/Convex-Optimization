import numpy as np
import matplotlib.pyplot as plt 
import cvxpy as cp

#%matplotlib qt

def DATA_gen():
    
    x = np.linspace(-10, 10, 30)
    
    y = x + np.random.normal(0, 2, len(x))
    data = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    data = np.vstack((data, np.array([-10, 10])))
    data = np.vstack((data, np.array([10, -10])))

    return data




def cvx_opt(penalty):

    m = cp.Variable(1)
    b = cp.Variable(1)

    if penalty == 'l2':
        cost = cp.norm(data[:,1] - (m*data[:,0] + b), 2)

    if penalty == 'l1':
        cost = cp.norm(data[:,1] - (m*data[:,0] + b), 1)

    if penalty == 'huber':
        cost = cp.sum(cp.huber(data[:,1] - (m*data[:,0] + b), M = 4))
    
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    
    b = b.value
    m = m.value
    
    return m, b




data = DATA_gen()

penalty = 'l2'
m2, b2 = cvx_opt(penalty)

penalty = 'l1'
m1, b1 = cvx_opt(penalty)

penalty = 'huber'
mh, bh = cvx_opt(penalty)


x = np.linspace(-10, 10, 100)
plt.plot(x, m2*x+b2, 'k--', label = 'l2 norm')
plt.plot(x, m1*x+b1, label = 'l1 norm')
plt.plot(x, mh*x+bh, label = 'huber')
plt.plot(data[:,0], data[:,1], 'ko', mfc = 'none')
plt.legend()

plt.title('Robust regression using the l1, l2, and huber penalty')

plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
plt.grid()
plt.show()