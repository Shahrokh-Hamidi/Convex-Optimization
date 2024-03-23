import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import cvxpy as cp
import scipy.io

#%matplotlib qt



X = np.array([[-2.95855623e-02,  1.95754105e-01,  9.38421496e-02,
        -3.29467959e-02, -1.73989577e-01, -7.97793709e-01,
        -1.11389191e-01, -7.42729742e-01, -1.29151943e-01,
        -5.17979960e-01, -1.09758793e-01, -2.30802197e-01,
        -7.73237398e-03,  7.09756728e-02, -3.52385404e-01,
        -5.31747802e-01, -3.04772028e-01, -1.29998958e-01,
        -4.48261427e-02, -3.27872167e-03, -4.41183114e-01,
        -4.05461983e-01, -1.99804573e-01,  4.26455138e-03,
         2.62474869e-03, -3.51403372e-01, -3.96796643e-01,
        -2.97028945e-01, -1.39562473e-01, -5.57034030e-01,
        -4.39141850e-01,  8.26766799e-04, -8.98575132e-02,
         1.98784110e-01,  1.48327407e-01, -7.65239272e-01,
        -1.64675275e-01,  1.46751529e-01, -6.94701156e-02,
        -2.10891761e-01, -4.41468102e-01, -3.05412384e-01,
        -4.84305217e-02, -2.79176289e-01, -5.10060070e-01,
        -7.98311175e-02, -3.10935435e-01, -3.72311823e-01,
         2.19521351e-01, -4.94892629e-01,  1.41049092e-01,
        -3.87453856e-01,  1.34704940e-01, -3.12572867e-01,
        -6.26808596e-01, -3.25919656e-01, -6.47490319e-01,
        -1.60047770e-01, -1.02291418e-01, -2.23325151e-01,
         1.01700226e-01, -5.31054659e-01, -4.42743514e-01,
         3.72895510e-01, -2.93658107e-02],
       [-3.27235847e-01,  4.95076766e-01,  5.29947667e-02,
        -9.59639561e-03,  7.63301831e-01, -2.65502307e-01,
        -9.16862531e-02,  2.55875621e-01, -1.70749169e-01,
         2.63716029e-01,  8.63248949e-01, -5.52139923e-01,
         4.47627192e-03, -1.00834217e-01, -5.59729838e-01,
         3.16068644e-01,  6.43951955e-02,  7.36883561e-02,
         1.32683793e-01,  3.80192236e-01, -5.87038834e-01,
         8.70099909e-02,  3.12597219e-01,  2.65889204e-03,
         4.41683197e-02,  5.15480333e-01, -4.29987771e-01,
        -3.98981772e-01,  1.11941884e-01, -7.95167137e-02,
         4.92736689e-01, -8.85028346e-03, -3.67032553e-01,
         6.48520918e-01,  6.98918608e-01, -3.16115610e-01,
         2.87103155e-01, -6.41579927e-01, -1.60079670e-01,
        -5.80573323e-03,  2.21465550e-01,  7.80819944e-01,
         3.49373009e-01, -2.23253724e-01,  2.95930684e-01,
        -7.49587566e-01, -1.23183214e-01, -8.86820600e-02,
        -4.88104745e-01,  1.19707947e-01, -6.29274097e-01,
        -2.48284959e-01, -6.85654170e-01,  3.10257939e-01,
        -6.87885168e-02, -8.22196336e-01,  3.28320725e-01,
         4.06064597e-01, -1.32930751e-02, -2.41369680e-01,
        -7.44523222e-02, -5.16360847e-01, -4.14590950e-01,
        -8.18568152e-01,  4.40919531e-02]])


Y = np.array([[-1.10929648, -1.19714993, -1.46155265,  0.54850428,  1.35098877,
        -0.63978276,  1.02322679,  1.52509307,  0.61292653,  2.04638631,
         1.82855895,  1.47921071,  0.30167283,  1.73791846,  0.7515328 ,
        -1.25628825, -0.88484211,  0.53786931,  1.53502706,  1.18898028,
         0.91609724, -1.54078336,  0.16536052, -1.19906352, -1.46105915,
         1.27808057, -1.12648287,  0.3258603 ,  1.14215806,  0.28618332,
         1.47693679,  0.89203912, -0.68661538, -1.388819  ,  1.73895725,
         1.57556233,  1.28373784, -0.87911447,  1.40727042, -1.40937883,
        -1.28074118, -1.48632542,  0.38125655, -1.52045199, -1.66213125,
        -1.94551437,  0.21882479,  0.11292287, -1.41478813, -1.87993473,
        -1.00727723,  0.02675306, -0.50369109, -1.39723257, -1.22424294,
         1.30030558, -1.01635215,  1.0255251 , -0.08169464, -0.25732126,
         1.40042328, -2.00438398,  1.36791438, -0.89550151, -1.55659656,
        -0.70043184,  0.99463546,  1.61911896, -1.79102261,  0.82796362,
        -1.17658101,  0.07050389,  1.21510545, -0.26214263,  1.28184009,
        -1.80832372,  1.53041901, -0.66284338, -1.14343021,  0.8866055 ,
        -1.22024598, -0.86028255,  0.44238846, -0.39174046, -0.33583426,
        -1.0828518 ,  0.22247235,  0.06527953,  1.66426983,  0.95034167,
         0.29308652, -0.70090159, -0.7519563 , -1.04020311, -1.73012502,
         1.17321142, -0.01267105,  0.44442947,  1.43795889, -0.96444755,
        -1.61170243,  1.07293517, -1.390684  ,  1.86297062, -1.17721976,
        -0.85478808,  0.94232596,  0.90095948,  0.78233839,  0.6769013 ,
         0.57207325,  1.07070613,  0.94326805, -0.88960536, -0.70306538,
         0.49395402, -0.72786602, -1.88687969, -0.32508892,  1.30512574,
         0.33981014,  0.30344273,  0.23495848,  0.34879119,  0.29813592,
         0.7812839 ,  0.61364035,  0.38713553,  0.75876337,  0.70237945,
         0.39402689,  0.31522323,  0.39765858,  0.5236989 ,  0.74799895,
         0.28614548,  0.24846108,  0.64083345,  0.49758925,  0.48985264,
         0.28120179,  0.42798242,  0.58821225,  0.41714166,  0.30514283,
         0.62799254],
       [ 0.3852498 ,  0.48917503,  0.23041757,  1.26242449,  1.2942288 ,
         1.26090287, -1.40510591,  1.42402683, -1.4810577 ,  0.07343144,
        -0.61127391, -1.37203234, -1.17497071, -0.79741137, -1.8623418 ,
        -0.01485873,  0.84298797, -1.78331797,  0.98114637,  1.2863001 ,
        -1.61188272,  0.85716211, -1.97658857, -0.6671302 , -0.41711143,
         0.30076968, -0.14360928, -1.13532496,  1.58119757, -1.41148329,
         0.52105578, -1.80541092, -1.13739852,  1.38593001, -0.24846801,
        -1.21754355,  0.93499571,  1.89903227,  0.43621015, -0.82161215,
        -0.03838718, -0.59754699, -1.47367586,  0.88731639,  0.61900299,
         0.66901519, -1.27319787,  1.205941  , -0.87794017, -0.86446196,
        -0.49818959,  1.97003418, -1.00804074, -0.81891045,  0.4138192 ,
        -1.26725377,  0.88957589,  1.75638958,  1.23448912,  1.60107613,
        -1.42114194,  0.3921614 ,  0.43390617, -1.24962578, -0.21339644,
        -1.03449096,  0.73352648,  0.20857677,  0.35460486,  1.2492346 ,
         0.86176662,  1.3834841 , -1.54888459, -1.70639443, -0.39449858,
         1.02390852, -0.82895152,  1.15288164,  1.36591446,  1.52710936,
        -0.1884893 ,  0.72104447, -1.28836322,  1.15171548, -1.12005379,
        -1.62522906, -1.26085395, -1.13053544, -0.77038397, -1.33230553,
         1.34445451,  1.29038998, -0.8204131 ,  1.69555841, -0.93375822,
        -0.67643161, -1.66943251,  1.17823186,  0.89618357, -1.0573024 ,
        -0.7036574 ,  1.64458793,  0.92030731,  0.92614406,  1.48093904,
         0.9156026 , -1.68778436,  0.92584162, -1.55306823, -1.63437019,
         1.7825389 ,  1.02149349, -1.21931587,  1.30945838,  1.55670293,
        -1.17650937, -0.82744872,  0.11239135,  1.58052194,  0.14566471,
        -0.20296804, -0.03981107,  0.15737806, -0.06000793,  0.28542558,
         0.05675917, -0.40796816,  0.00328204, -0.25757154,  0.31645786,
         0.1985505 , -0.26469693,  0.07997021, -0.61982956,  0.47241106,
        -0.21855985, -0.22959073,  0.40903737, -0.64462399,  0.28872615,
        -0.10924733, -0.406085  ,  0.2233023 ,  0.4763915 ,  0.40876068,
        -0.59554049]])


m = 2
P = cp.Variable((m,m), PSD = True)
q = cp.Variable(m)
r = cp.Variable(1)


cost = 0

constr = []
for i in range(X.shape[-1]):
    constr += [(X[:,i].reshape(1,-1)@P)@X[:,i].reshape(-1,1) + q.T@X[:,i].reshape(-1,1) + r <= -1]
    constr += [(Y[:,i].reshape(1,-1)@P)@Y[:,i].reshape(-1,1) + q.T@Y[:,i].reshape(-1,1) + r >= 1]

prob = cp.Problem(cp.Minimize(cost), constr) 
prob.solve(verbose = True)



P = P.value
q = q.value
r = r.value

points = np.vstack((np.linspace(-1,1,100).reshape(1,-1), np.linspace(-1,1,100).reshape(1,-1)))
discr = (points.T@P)@points + points.T@q.reshape(-1,1) + r


#plt.plot(X[0,:], X[1,:], 'ro')
#plt.plot(Y[0,:], Y[1,:], 'go')