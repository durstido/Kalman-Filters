import numpy as np
import matplotlib.pyplot as plt

n_iter = 10 #50 initial parameters
sz = (n_iter,) #size of array
x = -0.37727 #truth value
zNOT = np.random.normal(x, 0.1, size = sz) #measurments that are returned from sensors at each location, observations
                                        #output you can meausre//y

z1=[.39, .50, .48, .29, .25, .32, .34, .48, .41, .45] #GPS
z2=[.42, .51, .20, .30, .45, .

#allocate space for arrays
#priori:
xhatminus=np.zeros(sz)
pminus=np.zeros(sz)

#posteri:
xhat=np.zeros(sz) #estimate of x
p= np.zeros(sz) #error estimate covariance

#Kalman gain
K=np.zeros(sz)

#variances
Q = 1e-5 #for predicition pminus
R=.1 #estimate of measurement variance, in Kalman Gain

#initial guess
xhat[0]=0
p[0]=1

for k in range(1, n_iter):
    #step 1: prediction
    xhatminus[k]=xhat[k-1]+Q
    pminus[k]=p[k-1]

    #step 2: update
    K[k]=pminus[k]/(pminus[k]+R)
    xhat[k]=xhatminus[k]+K[k]*(z1[k]-xhatminus[k])
    p[k]=(1-K[k])*pminus[k]

    print("xhat #{}: {}").format(k,xhat[k])
    print("p #{}: {}").format(k,p[k])


plt.figure()
plt.plot(z,'k+',label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
plt.axhline(x,color='g',label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')

plt.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
plt.plot(valid_iter,pminus[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()
