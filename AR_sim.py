#ARSIM	Simulation of AR process.	

# v=ARSIM(w,A,C,n) simulates n time steps of the AR(p) process

#     v(k,:)' = w' + A1*v(k-1,:)' +...+ Ap*v(k-p,:)' + eta(k,:)', 

#  where A=[A1 ... Ap] is the coefficient matrix, and w is a vector of
#  intercept terms that is included to allow for a nonzero mean of the
#  process. The vectors eta(k,:) are independent Gaussian noise
#  vectors with mean zero and covariance matrix C.

#  The p vectors of initial values for the simulation are taken to
#  be equal to the mean value of the process. (The process mean is
#  calculated from the parameters A and w.) To avoid spin-up effects,
#  the first 10^3 time steps are discarded. Alternatively,
#  ARSIM(w,A,C,n,ndisc) discards the first ndisc time steps.
#
#  ARSIM(w,A,C,[n, ntr]) generates ntr realizations (trials) of
#  length n of the AR(p) process, which are output as the matrices
#  v(:,:,itr) with itr=1,...,ntr. 

# TODO: only for now to build
import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

w  = np.array([ 0.25,  0.1 ]).T
A1 = np.array([ [ 0.4 , 1.2], [0.3  ,0.7] ])
A2 = np.array([ [ 0.35, -0.3], [-0.4 ,-0.5]])

A = np.concatenate((A1,A2),axis=1)

C = np.array([[1.00 , 0.50],[0.50 ,1.50]])
ndisc = 10**3
n = 200
ntr = 5


m       = C.shape[0]        # dimension of state vectors 
p       = int(A.shape[1] / m)    # order of process
n       = n                 # number of time steps

# TODO: add input checks later

w = w.flatten() # make sure w is a column vector

#Check whether specified model is stable


m1 = np.concatenate((np.eye((p-1)*m),np.zeros(((p-1)*m,m))),axis=1)
A1 = np.concatenate((A,m1),axis=0)

eigval, v = np.linalg.eig(A1)


if np.any(abs(eigval) > 1):
    raiseValueError('The specified AR model is not stable')
    # TODO: add return later

# TODO: @ Gregor I can check if all eigenvalues of matrix are positive, right?
if not np.all(np.linalg.eigvals(C) > 0):
    raiseValueError('The covariance matrix C must be positive definite')

# Choletzky decomposition of C
R = cholesky(C,lower=False) # Upper triangular matrix

# Get ntr realizations of ndisc+n independent Gaussian pseudo-random vectors with covariance matrix C=R'*R
randvec = np.empty((ndisc+n,m,ntr))
for iter in range(ntr):
    randvec[:,:,iter] = np.random.normal(size=(ndisc+n,m)) @ R
  
# add the intercept vector
ww = w[:,np.newaxis] # make sure its the same as matlab version
randvec += np.tile(ww,[ndisc+n,1, ntr])

# Get transpose of system matrix A (use transpose in simulation because 
# we want to obtain the states as row vectors)
AT      = A.T

# Take the p initial values of the simulation to equal the process mean, 
# which is calculated from the parameters A and w
if np.any(w):
    B = np.eye((m))

    for j in range(p):
        B -= A[:, ((j)*m+1)-1:(j+1)*m]

    mval = np.linalg.solve(B,w)

    x = np.tile(mval,[p,1])

else:
    x = np.zeros((p,m))

# Initialize state vectors
mat1 = np.concatenate((x,np.zeros((ndisc+n,m))),axis=0)
u    = np.tile(mat1[:,:,np.newaxis],[1,1,ntr])


# Simulate ntr realizations of n+ndisc time steps. In order to be
# able to make use of Matlab's vectorization capabilities, the
# cases p=1 and p>1 must be treated separately.



for itr in range(ntr):
    for k in range(p+1,ndisc+n+p+1):
        for j in range(p):
            x[j-1,:] = u[k-j-1,:,itr] @ AT[(j*m+1)-1:(j+1)*m,:]
        u[k-1,:,itr] = np.sum(x,axis=0) + randvec[(k-1)-p,:,itr]

    #u[k,:,itr] = np.sum(x,axis=1) + randvec[(k-1)-p,:,itr]
