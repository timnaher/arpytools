# %%
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def arfit(v,pmin,pmax,selector='sbc',no_const=False):
    """
    This function fits a polynomial to a set of data points.
    The polynomial is defined by the number of parameters pmin to pmax."""
    # n:   number of time steps (per realization)
    # m:   number of variables (dimension of state vectors) 
    # ntr: number of realizations (trials)

    n,m,ntr = v.shape

    # input checks
    # TODO: is this a good way of checking the input?
    if (not isinstance(pmin, int)) | (not isinstance(pmax, int)):
        raise ValueError("Orders must be integers")

    if (pmax < pmin):
        raise ValueError('PMAX must be greater than or equal to PMIN.')

    mcor = 0 if no_const == True else 1 # fit the constant if mcor = 1


    ne      = ntr*(n-pmax) # number of block equations of size m
    npmax	= m*pmax+mcor # maximum number of parameter vectors of length m


    if (ne <= npmax):
        raise ValueError('Time series too short.')

    # compute QR factorization for model of order pmax
    R, scale   = arqr(v, pmax, mcor)

    #TODO: for now we take the inpuit order as the maximum order. In the future we should include the search through the orders

    # temporary
    popt         = pmax
    nnp           = m*popt + mcor # number of parameter vectors of length m


    # decompose R for the optimal model order popt according to 
    #
    #   | R11  R12 |
    # R=|          |
    #   | 0    R22 |
    #

    R11   = R[0:nnp, 0:nnp]
    R12   = R[0:nnp, (npmax+1)-1:npmax+m]
    R22   = R[(nnp+1)-1:npmax+m, (npmax+1)-1:npmax+m]

    if (nnp>0):
        if (mcor == 1):
            # improve condition of R11 by rescaling the first column
            con = np.max(scale[1:npmax+m]) / scale[0]
            R11[0,0] = R11[0,0] * con
        Aaug = scipy.linalg.solve(R11, R12).T

        # return coefficint matrix A and intercept vector w separately
        if (mcor == 1):
            # intercept vector w is the first column of Aaug, rest of Aaug is the coefficient matrix A
            w = Aaug[:,0] * con
            A = Aaug[:,1:nnp]

        else:
            # return intercept vector of zeros
            w = np.zeros((m,1))
            A = Aaug

    else:
        # no parameters have estimated
        # return only covariance matrix estimate and order selection criterion
        w = np.zeros((m,1))
        A = []

    # return covariance matrix
    dof = ne - nnp  # number of block degrees of freedom
    C   = R22.T * R22/dof # bias-corrected estimate of covariance matrix

    invR11 = np.linalg.inv(R11)

    if (mcor == 1):
        # undo condition improving scaling
        invR11[0, :] = invR11[0, :] * con

    Uinv   = invR11*invR11.T
    frow   = np.concatenate([np.array([dof]), np.zeros((Uinv.shape[1]-1))], axis=0)
    th     = np.vstack((frow,Uinv))

    return w, A, C, th




def arqr(v, p, mcor):
    n,m,ntr = v.shape
    ne     = ntr*(n-p)  # number of block equations of size m
    nnp    = m*p+mcor   # number of parameter vectors of size m

    # init K
    K = np.zeros((ne,nnp+m))     

    if mcor == 1:
        K[:,0] = np.squeeze(np.ones((ne,1))) #TODO: find a better way to do this

    # build K
    for itr in range(1,ntr+1):
        for j in range(1,p+1):
            myarr = np.squeeze(v[(p-j+1)-1 : (n-j), :, (itr)-1]) # changes the indexing from python to matlab
            K[ ((n-p)*(itr-1) + 1)-1 : ((n-p)*itr), (mcor+m*(j-1)+1)-1 : (mcor+m*j)] = myarr.reshape((myarr.shape[0],1)) # TODO: check if this is correct
        
        myarr2 = np.squeeze(v[ (p+1)-1:n,:,itr-1 ])
        K[ ((n-p)*(itr-1) + 1)-1 : ((n-p)*itr), (nnp+1)-1 : nnp+m ] = myarr2.reshape((myarr2.shape[0],1))

    q     = nnp + m  # number of columns of K
    # times epsilon as floating point number precision
    delta = (q**2 + q + 1) * np.finfo(np.float64).eps # Higham's choice for a Cholesky factorization
    scale = np.sqrt(delta) * np.sqrt( np.sum(K**2,axis=0))
    Q, R  = scipy.linalg.qr(np.vstack((K,np.diag(scale))))

    return  np.triu(R), scale


def ar_interp(v,A,extrasamp,Fs):
    Fs = 1000
    origsamps = len(v) # Number of samples in the to-be-extrapolated signal
    arord     = A.shape[1] # Order of AR

    nan_array    = np.empty((extrasamp))
    nan_array[:] = np.nan
    exdat        = np.concatenate([ v[:,0,0] , nan_array], axis=0) # add nan's to the end of the original signal to future extrapolated samples

    for es in np.arange(extrasamp):
        currsamp = origsamps+es; # Location of new sample in the vector
        # For a n order AR model a with noise variance c, value x at time t is given by the
        # following equation : x(t) = a(1)*x(t-1) + a(2)*x(t-2) + ... +
        # a(n-1)*x(t-n+1) + a(n)*x(t-n) + sqrt(c)*randnoise

        # extrapolate the signal
        exdat[currsamp] = np.sum(A * np.flip(exdat[(currsamp-arord) : (currsamp)])) + np.sqrt(np.abs(C))*np.random.randn(1,1)

    return exdat

# %%


time = np.arange(0, 2, 0.01)
v1    = np.sin(2*np.pi*4*time) #+ np.random.rand(len(time))/10
v2   = np.cos(2*np.pi*4*time) #+ np.random.rand(len(time))
#v3   = np.cos(2*np.pi*7*time) + np.random.rand(len(time))


v    = np.vstack((v1,v1)).T
v    = v.reshape((v.shape[0],1,2))

w, A, C, th = arfit(v,1,20,selector='sbc',no_const=False)



Fs        = 1000
extrams   = 1000
origsamps = len(v) # Number of samples in the to-be-extrapolated signal
arord     = A.shape[1] # Order of AR

nan_array    = np.empty((extrams))
nan_array[:] = np.nan
exdat        = np.concatenate([ v[:,0,0] , nan_array], axis=0) # add nan's to the end of the original signal to future extrapolated samples
for es in np.arange(extrams):
    currsamp = origsamps+es; # Location of new sample in the vector
    # For a n order AR model a with noise variance c, value x at time t is given by the
    # following equation : x(t) = a(1)*x(t-1) + a(2)*x(t-2) + ... +
    # a(n-1)*x(t-n+1) + a(n)*x(t-n) + sqrt(c)*randnoise

    # extrapolate the signal
    exdat[currsamp] = np.sum(A * np.flip(exdat[(currsamp-arord) : (currsamp)])) + np.sqrt(np.abs(C)) * np.random.randn(1,1)


fig, ax = plt.subplots(3,1,figsize = (8,10),constrained_layout=True)
fig.tight_layout()
plt.style.use('dark_background')
ax[0].plot(v[:,0,0])
ax[0].plot(v[:,0,1])
#ax[0].plot(v[:,0,2])

ax[0].set_title('Original Signal')
ax[1].plot(A[0,:])
ax[1].set_title('AR Coefficients')
ax[2].plot(np.arange(200),exdat[:origsamps])
ax[2].plot(np.arange(200,200+extrams),exdat[origsamps:])
ax[2].set_title('Extrapolated Signal')
ax[2].set_xlabel('Time (ms)')
ax[2].legend(['Original','Extrapolated'])








# %%

v = np.random.rand(10,3,2)
import matplotlib.pyplot as plt
import numpy as np

time  = np.arange(0, 2, 0.01);
v     = np.sin(2*np.pi*4*time)
v     = v.reshape((v.shape[0],1,1))

#v = np.random.rand(10,1,2)
pmax    = 10
mcor = 1

"""
This function fits a polynomial to a set of data points.
The polynomial is defined by the number of parameters pmin to pmax."""
# n:   number of time steps (per realization)
# m:   number of variables (dimension of state vectors) 
# ntr: number of realizations (trials)
n,m,ntr = v.shape

#TODO: include input check and set defaults













mcor     = 1 # fit the intercept vector
selector = 'sbc' # use sbc as order selection criterion

ne      = ntr*(n-pmax);         # number of block equations of size m
npmax	= m*pmax+mcor;          # maximum number of parameter vectors of length m

R, scale   = arqr(v, pmax, mcor)

#TODO: for now we take the inpuit order as the maximum order. In the future we should include the search through the orders

# temporary
popt         = pmax
nnp           = m*popt + mcor # number of parameter vectors of length m


# decompose R for the optimal model order popt according to 
#
#   | R11  R12 |
# R=|          |
#   | 0    R22 |
#

R11   = R[0:nnp, 0:nnp]
R12   = R[0:nnp, (npmax+1)-1:npmax+m]    
R22   = R[(nnp+1)-1:npmax+m, (npmax+1)-1:npmax+m]

if (nnp>0):
    if (mcor == 1):
        # improve condition of R11 by rescaling the first column
        con = np.max(scale[1:npmax+m]) / scale[0]
        R11[0,0] = R11[0,0] * con
    Aaug = scipy.linalg.solve(R11, R12).T

    # return coefficint matrix A and intercept vector w separately
    if (mcor == 1):
        # intercept vector w is the first column of Aaug, rest of Aaug is the coefficient matrix A
        w = Aaug[:,0] * con
        A = Aaug[:,1:nnp]

    else:
        # return intercept vector of zeros
        w = np.zeros((m,1))
        A = Aaug

else:
    # no parameters have estimated
    # return only covariance matrix estimate and order selection criterion
    w = np.zeros((m,1))
    A = []

# return covariance matrix
dof = ne - nnp  # number of block degrees of freedom
C   = R22.T * R22/dof # bias-corrected estimate of covariance matrix

invR11 = np.linalg.inv(R11)

if (mcor == 1):
    # undo condition improving scaling
    invR11[0, :] = invR11[0, :] * con

Uinv   = invR11*invR11.T
frow   = np.concatenate([np.array([dof]), np.zeros((Uinv.shape[1]-1))], axis=0)
th     = np.vstack((frow,Uinv))

    

fig, ax = plt.subplots(2,1)
ax[0].plot(time,v[:,0,0])
ax[1].plot(A[0,:])

# %%
