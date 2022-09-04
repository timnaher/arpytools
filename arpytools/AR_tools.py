# %%
import numpy as np
import numpy.linalg as la
import scipy.linalg 
import matplotlib.pyplot as plt

def arfit(v,pmin,pmax,selector='sbc',no_const=False):
    """ This function fits a polynomial to a set of data points.
    The polynomial is defined by the number of parameters pmin to pmax

    :param v: time series data (time x variables x trials)
    :type v: numpy array
    :param pmin: minimum order of the polynomial which to check
    :type pmin: int
    :param pmax: maximum order of the polynomial which to check
    :type pmax: int
    :param selector: which parameter to evaluate to find the best model order.
        Options are Schwarz-Bayes Criterior and Akaikes Final Prediction error, defaults to 'sbc'
    :type selector: str, optional
    :param no_const: If True, no intercept is fit in the model, defaults to False
    :type no_const: bool, optional
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: w, A, C, th. w is the estimated parameters, A is the estimated AR coefficients, C is the estimated noise variance, th is the estimated threshold
    :rtype: np.array, np.array, np.array, np.array
    """   

    n,m,ntr = v.shape # dimensions of v

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

    sbc, fpe, _ , _ = arord(R, m, mcor, ne, pmin, pmax)

    # get index iopt of order that minimizes the order selection criterion specified by the variable selector
    if selector == 'sbc':
        popt = pmin + np.argmin(sbc)
    elif selector == 'fpe':
        popt = pmin + np.argmin(fpe)
    nnp = m*popt + mcor # number of parameter vectors of length m
    # decompose R for the optimal model order popt according to 
    #     | R11  R12 |
    # R = |          |
    #     | 0    R22 |

    R11   = R[0:nnp, 0:nnp]
    R12   = R[0:nnp, (npmax+1)-1:npmax+m]
    R22   = R[(nnp+1)-1:npmax+m, (npmax+1)-1:npmax+m]

    if (nnp>0):
        if (mcor == 1):
            # improve condition of R11 by rescaling the first column
            con = np.max(scale[1:npmax+m]) / scale[0]
            R11[:,0] = R11[:,0] * con
        Aaug = la.solve(R11, R12).T

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
        w = np.zeros((m,1)) # return only covariance matrix estimate and order selection criterion

        A = []

    # return covariance matrix
    dof = ne - nnp  # number of block degrees of freedom
    C   = R22.T @ R22/dof # bias-corrected estimate of covariance matrix
    invR11 = np.linalg.inv(R11)

    if (mcor == 1):
        # undo condition improving scaling
        invR11[0, :] = invR11[0, :] * con

    Uinv   = invR11@invR11.T
    frow   = np.concatenate([np.array([dof]), np.zeros((Uinv.shape[1]-1))], axis=0)
    th     = np.vstack((frow,Uinv))
    return w, A, C, th


def arqr(v, p, mcor):
    """ QR decomposition 

    :param v: time series data (time x variables x trials)
    :type v: numpy array
    :param p: maximum order of the polynomial (pmax from arfit)
    :type p: int
    :param mcor: if True, no intercept is fit in the model, defaults to False
    :type mcor: bool, optional
    :return: upper triangular matrix R and scale vector
    :rtype: _type_
    """    

    n,m,ntr = v.shape # dimensions of v
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
            K[ ((n-p)*(itr-1) + 1)-1 : ((n-p)*itr), (mcor+m*(j-1)+1)-1 : (mcor+m*j)] = myarr# myarr.reshape((myarr.shape[0],1)) # TODO: check if this is correct

        myarr2 = np.squeeze(v[ (p+1)-1:n,:,itr-1 ])
        K[ ((n-p)*(itr-1) + 1)-1 : ((n-p)*itr), (nnp+1)-1 : nnp+m ] = myarr2#.reshape((myarr2.shape[0],1))

    q     = nnp + m  # number of columns of K
    # times epsilon as floating point number precision
    delta = (q**2 + q + 1) * np.finfo(np.float64).eps # Higham's choice for a Cholesky factorization
    scale = np.sqrt(delta) * np.sqrt( np.sum(K**2,axis=0))
    Q, R  = la.qr(np.vstack((K,np.diag(scale))))

    return  R, scale

def arord(R, m, mcor, ne, pmin, pmax):
    """_summary_

    :param R: R matrix from QR decomposition
    :type R: numpy array
    :param m: number of variables
    :type m: int
    :param mcor:  if True, no intercept is fit in the model, defaults to False
    :type mcor: bool, optional
    :param ne: number of block equations of size m
    :type ne: int
    :param pmin: minimum order of the polynomial (pmin from arfit)
    :type pmin: int
    :param pmax: maximum order of the polynomial (pmax from arfit)
    :type pmax: int
    :return: sbc, fpe, logdp, num_p
    :rtype: _type_
    """    

    imax 	  = pmax-pmin+1 # maximum index of output vectors

    # initialize output vectors
    sbc     = np.zeros((imax))     # Schwarz's Bayesian Criterion
    fpe     = np.zeros((imax))     # log of Akaike's Final Prediction Error
    logdp   = np.zeros((imax))     # determinant of (scaled) covariance matrix
    num_p   = np.zeros((imax))     # number of parameter vectors of length m

    num_p[imax-1]= m*pmax+mcor

    #Get lower right triangle R22 of R: 
    #    | R11  R12 |
    # R= |          |
    #    | 0    R22 |

    R22     = R[int(num_p[imax-1]+1)-1 : int(num_p[imax-1]+m), int(num_p[imax-1]+1)-1 : int(num_p[imax-1]+m)]
    # From R22, get inverse of residual cross-product matrix for model of order pmax
    invR22  = np.linalg.inv(R22) # TODO: this is slights different from MATLAB
    Mp      = invR22@invR22.T

    # For order selection, get determinant of residual cross-product matrix
    logdp[imax-1] = 2 * np.log(np.abs(np.prod( np.diag(R22) )))
    # Compute approximate order selection criteria for models of  order pmin:pmax
    i = imax

    for p in np.arange(pmax,pmin-1,-1) :
        num_p[i-1]   = m*p + mcor	# number of parameter vectors of length m

        if p < pmax:
            # Downdate determinant of residual cross-product matrix
            # Rp: Part of R to be added to Cholesky factor of covariance matrix
            Rp       = R[int(num_p[i-1]+1)-1 : int(num_p[i-1]+m), int(num_p[imax-1]+1)-1:int(num_p[imax-1]+m)]

            # Get Mp, the downdated inverse of the residual cross-product matrix, using the Woodbury formula
            L        = np.linalg.cholesky(np.eye(m) + Rp@Mp@Rp.T).T
            N        = scipy.linalg.solve(L,Rp*Mp)
            Mp       = Mp - N.T*N

            # Get downdated logarithm of determinant
            logdp[i-1] = logdp[i] + 2*np.log(np.abs(np.prod(np.diag(L))))
        # Schwarz's Bayesian Criterion
        sbc[i-1] = logdp[i-1]/m - np.log(ne) * (ne-num_p[i-1])/ne

        # logarithm of Akaike's Final Prediction Error
        fpe[i-1] = logdp[i-1]/m - np.log(ne*(ne-num_p[i-1])/(ne+num_p[i-1]))
        i -= 1

    return sbc, fpe, logdp, num_p



def ar_interp(v,A,extrasamp,Fs):
    """ Interpolate data based on the esimated AR model

    :param v: time series data (n x m x ntr)
    :type v: numpy array
    :param A: estimated AR model (p x m x m x ntr)
    :type A: numpy array
    :param extrasamp: number of samples to interpolate
    :type extrasamp: int
    :param Fs: sampling frequency
    :type Fs: int
    :return: _description_
    :rtype: _type_
    """    
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


def adjph(x): 
    """Given a complex matrix X, OX=ADJPH(X) returns the complex matrix OX
    #  that is obtained from X by multiplying column vectors of X with
    #  phase factors exp(i*phi) such that the real part and the imaginary
    #  part of each column vector of OX are orthogonal and the norm of the
    #  real part is greater than or equal to the norm of the imaginary
    #  part.
    #  ADJPH is called by ARMODE.
    #  Reference:
    #  P.J. Brockwell and R.A. Davis, "Orthogonal Array Analysis,
    #  Second Edition", Prentice-Hall, pp. 30-31, 2000.

    :param x: complex N x M matrix
    :type x: numpy array
    :return: OX such the real part of each column vector of OX is orthogonal and the norm of the real part is greater than or equal to the norm of the imaginary part
    :rtype: _type_
    """    

    #  Check input
    [n,m] = x.shape
    ox    = np.zeros((n,m),dtype=complex)
    for jj in range(m):
        a       = np.real(x[:,jj])
        b       = np.imag(x[:,jj])
        phi     = 0.5 * np.arctan( 2 * np.sum(a*b) / (b.T @ b - a.T @ a) )
        bnorm   = np.linalg.norm(np.sin(phi) * a+np.cos(phi) * b)    # norm of new imaginary part
        anorm   = np.linalg.norm(np.cos(phi) * a+np.sin(phi) * b)    # norm of new real part
        if bnorm > anorm:
            phi = phi - np.pi/2 if phi < 0 else phi + np.pi/2
        ox[:,jj] = x[:,jj] * np.exp(1j*phi)
    return ox

