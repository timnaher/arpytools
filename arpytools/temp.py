# %%
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def arqr(v, p, mcor):
    n, m, ntr = v.shape
    ne = ntr * (n - p)  # number of block equations of size m
    nnp = m * p + mcor  # number of parameter vectors of size m

    # init K
    K = np.zeros((ne, nnp + m))

    if mcor == 1:
        K[:, 0] = np.squeeze(np.ones((ne, 1)))  # TODO: find a better way to do this

    # build K
    for itr in range(1, ntr + 1):
        for j in range(1, p + 1):
            myarr = np.squeeze(
                v[(p - j + 1) - 1 : (n - j), :, (itr) - 1]
            )  # changes the indexing from python to matlab
            K[
                ((n - p) * (itr - 1) + 1) - 1 : ((n - p) * itr),
                (mcor + m * (j - 1) + 1) - 1 : (mcor + m * j),
            ] = myarr.reshape(
                (myarr.shape[0], 1)
            )  # TODO: check if this is correct

        myarr2 = np.squeeze(v[(p + 1) - 1 : n, :, itr - 1])
        K[
            ((n - p) * (itr - 1) + 1) - 1 : ((n - p) * itr), (nnp + 1) - 1 : nnp + m
        ] = myarr2.reshape((myarr2.shape[0], 1))

    q = nnp + m  # number of columns of K
    # times epsilon as floating point number precision
    delta = (q ** 2 + q + 1) * np.finfo(
        np.float64
    ).eps  # Higham's choice for a Cholesky factorization
    scale = np.sqrt(delta) * np.sqrt(np.sum(K ** 2, axis=0))
    Q, R = scipy.linalg.qr(np.vstack((K, np.diag(scale))))

    return np.triu(R), scale


def arord(R, m, mcor, ne, pmin, pmax):

    imax = pmax - pmin + 1  # maximum index of output vectors

    # initialize output vectors
    sbc = np.zeros((imax))  # Schwarz's Bayesian Criterion
    fpe = np.zeros((imax))  # log of Akaike's Final Prediction Error
    logdp = np.zeros((imax))  # determinant of (scaled) covariance matrix
    num_p = np.zeros((imax))  # number of parameter vectors of length m

    num_p[imax - 1] = m * pmax + mcor

    # Get lower right triangle R22 of R:

    #   | R11  R12 |
    # R=|          |
    #   | 0    R22 |

    R22 = R[
        int(num_p[imax - 1] + 1) - 2 : int(num_p[imax - 1] + m) - 1,
        int(num_p[imax - 1] + 1) - 2 : int(num_p[imax - 1] + m) - 1,
    ]

    # From R22, get inverse of residual cross-product matrix for model of order pmax
    invR22 = np.linalg.inv(R22)  # TODO: this is slights different from MATLAB
    Mp = invR22 * invR22.T

    # For order selection, get determinant of residual cross-product matrix
    #       logdp = log det(residual cross-product matrix)

    logdp[imax - 1] = 2 * np.log(np.abs(np.prod(np.diag(R22))))

    # Compute approximate order selection criteria for models of  order pmin:pmax
    i = imax

    for p in np.arange(pmax, pmin - 1, -1):
        num_p[i - 1] = m * p + mcor  # number of parameter vectors of length m

        if p < pmax:
            # Downdate determinant of residual cross-product matrix
            # Rp: Part of R to be added to Cholesky factor of covariance matrix
            Rp = R[
                int(num_p[i - 1] + 1) - 2 : int(num_p[i - 1] + m) - 1,
                int(num_p[imax - 1] + 1) - 2 : int(num_p[imax - 1] + m) - 1,
            ]

            # Get Mp, the downdated inverse of the residual cross-product matrix, using the Woodbury formula
            L = np.linalg.cholesky(np.eye(m) + Rp * Mp * Rp.T).T

            N = scipy.linalg.solve(L, Rp * Mp)
            Mp = Mp - N.T * N

            # Get downdated logarithm of determinant
            logdp[i - 1] = logdp[i] + 2 * np.log(np.abs(np.prod(np.diag(L))))

        # Schwarz's Bayesian Criterion
        sbc[i - 1] = logdp[i - 1] / m - np.log(ne) * (ne - num_p[i - 1]) / ne

        # logarithm of Akaike's Final Prediction Error
        fpe[i - 1] = logdp[i - 1] / m - np.log(
            ne * (ne - num_p[i - 1]) / (ne + num_p[i - 1])
        )

        i -= 1

    return sbc, fpe, logdp, num_p


# %% Test
time = np.arange(0, 2, 0.01)
v1 = np.sin(2 * np.pi * 4 * time)  # + np.random.rand(len(time))/10
v2 = np.cos(2 * np.pi * 4 * time)  # + np.random.rand(len(time))
# v3   = np.cos(2*np.pi*7*time) + np.random.rand(len(time))


v = np.vstack((v1, v1)).T
v = v.reshape((v.shape[0], 1, 2))

pmin = 1
pmax = 20
selector = "sbc"
no_const = False
n, m, ntr = v.shape

# input checks
# TODO: is this a good way of checking the input?
if (not isinstance(pmin, int)) | (not isinstance(pmax, int)):
    raise ValueError("Orders must be integers")

if pmax < pmin:
    raise ValueError("PMAX must be greater than or equal to PMIN.")

mcor = 0 if no_const else 1


ne = ntr * (n - pmax)  # number of block equations of size m
npmax = m * pmax + mcor  # maximum number of parameter vectors of length m


if ne <= npmax:
    raise ValueError("Time series too short.")

# compute QR factorization for model of order pmax
R, scale = arqr(v, pmax, mcor)

sbc, fpe, _, _ = arord(R, m, mcor, ne, pmin, pmax)

if selector == "sbc":
    order = pmin + np.argmin(sbc) + 1
elif selector == "fpe":
    order = pmin + np.argmin(fpe) + 1


# get the index and value of the selector


# %%
