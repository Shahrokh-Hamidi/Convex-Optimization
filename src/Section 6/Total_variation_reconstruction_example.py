import numpy as np
import matplotlib.pyplot as plt 
import cvxpy as cp

#%matplotlib qt


n = 4000
t = np.arange(0, n)

signal = 0.5*np.sin(0.05*(2*np.pi/n)*t + np.pi/4)


for i in range(len(signal)):
    if n//4 <= i <= n//2:
        signal[i] = -signal[i] 

    if  3*n//4 <= i:
        signal[i] = - signal[i]
    
noisy_signal = signal + np.random.normal(0,0.1,n)


lambda_ = 1
D = np.zeros((n,n))

for i in range(n-1):
    D[i, i:i+2] = [-1,1]



x = cp.Variable(n)

cost = cp.norm(noisy_signal - x, 2) + lambda_*cp.norm(D@x,1)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

denoised_signal = x.value


plt.subplot(311)
plt.plot(t, signal, 'k', lw = 2,  label = 'signal')
plt.xlabel('x', fontsize = 12)
plt.ylabel('y', fontsize = 12)
plt.ylim(-0.6, 0.6)
ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
plt.grid()
plt.legend()
plt.title('Total variation reconstruction example', fontsize = 16)



plt.subplot(312)
plt.plot(t, noisy_signal, label = 'noisy_signal')
plt.xlabel('x', fontsize = 12)
plt.ylabel('y', fontsize = 12)
plt.ylim(-0.6, 0.6)
ax = plt.gca()
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
plt.grid()
plt.legend()



plt.subplot(313)
plt.plot(t, denoised_signal, 'r', label = 'denoised_signal')
plt.xlabel('x', fontsize = 12)
plt.ylabel('y', fontsize = 12)
ax = plt.gca()
plt.ylim(-0.6, 0.6)
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
plt.grid()
plt.legend()


plt.show()
