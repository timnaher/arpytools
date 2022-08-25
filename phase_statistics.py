# %%
from cProfile import label
import cmath
from curses.ascii import RS
from math import *

import colorednoise as cn
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as sig
from fooof import FOOOF
from scipy import stats
from scipy.fft import fft, fftshift
from scipy.signal import savgol_filter
import tqdm
from numba import njit, prange


plt.style.use('dark_background')

from plotting_tools import *
from sim_tools import *

# class definition
class PhaseSimulator:
    def __init__(self,oscillation_freq = 10,   # frequency of oscillation
                 FS                    = 1000, # sampling frequency
                 WIN_SIZE              = 100,  # Hann window size
                 window_stop           = 500,  # sample at which to stop window, i.e. critical time or time of event onset
                 PROP_LENGTH           = 500,  # length of phase propagation
                 eps                   = 0.1,  # phase diffusion strength
                 niter                 = 1000, # number of iterations to simulate experiment
                 random_start          = True, # if True, start at random phase to each iteration
                 pink_noise_level      = 0.5,  # pink noise level of lfp
                 Prop_taper_k          = 0,    # asymmetry of taper. posivie = right skewerd, negative = left skewed, 0 = no skew
                 AR_taper_k            = 10,    # asymmetry of taper. posivie = right skewerd, negative = left skewed, 0 = no skew              
                 nSamples              = 1000, # number of samples in LFP data
                 nChannels             = 1,   # number of LFP channels or alternatively LFP trials
                 t_before_ct           = 50,    # time before critical time to start phase propagation
                 AR_ORD                = 50): # order of AR model

        self.oscillation_freq  = oscillation_freq
        self.FS                = FS
        self.WIN_SIZE          = WIN_SIZE
        self.window_stop       = window_stop
        self.PROP_LENGTH       = PROP_LENGTH
        self.eps               = eps
        self.niter             = niter
        self.random_start      = random_start
        self.pink_noise_level  = pink_noise_level
        self.Prop_taper_k      = Prop_taper_k
        self.AR_taper_k        = AR_taper_k
        self.nSamples          = 1000
        self.nChannels         = nChannels
        self.AR_ORD            = AR_ORD
        self.t_before_ct       = t_before_ct

    def generate_lfp(self):
        """ generates lfp and ground truth phase data """        
        lfp, pdata = sim_lfp(self.oscillation_freq,nChannels=self.nChannels,
                             eps=self.eps,FS=self.FS,nSamples=self.nSamples,
                             pink_noise_level=self.pink_noise_level,
                             random_start=self.random_start)
        self.lfp   = lfp.T
        self.pdata = pdata

    def generate_gtruth(self):
        """ generates ground truth data """

        # set ground truth
        self.gtruth  = self.pdata[self.window_stop-self.t_before_ct : (self.window_stop-self.t_before_ct+self.PROP_LENGTH)]

    def propagate_phase(self):
        """ propagates the phase """
        _, taper = get_w_at(self.Prop_taper_k,self.WIN_SIZE) # generate a taper for the propagation. Note: this is not necessarily the same as the AR taper
        self.Prop_taper = taper # this is symmetric if k = 0, so its a hann, asymmetric if k != 0

        # init empty prop phase 
        self.prop_phase = np.empty((self.PROP_LENGTH,self.nChannels))

        if self.nChannels > 1: # loop over channels / trials
            for j in range(self.nChannels):
                prop_phase = propagate_phase(self.lfp[:,j],WIN_SIZE=self.WIN_SIZE,
                                    TARGET_FREQ=self.oscillation_freq,
                                    FS=self.FS,
                                    NPAD=1000,
                                    PROP_LENGTH=self.PROP_LENGTH,
                                    window_stop=self.window_stop,
                                    taper=self.Prop_taper,
                                    t_before_ct=self.t_before_ct)
                self.prop_phase[:,j]   = prop_phase
        else:
            prop_phase = propagate_phase(self.lfp,WIN_SIZE=self.WIN_SIZE,
                                    TARGET_FREQ=self.oscillation_freq,
                                    FS=self.FS,
                                    NPAD=1000,
                                    PROP_LENGTH=self.PROP_LENGTH,
                                    window_stop=self.window_stop,
                                    taper=self.Prop_taper,
                                    t_before_ct=self.t_before_ct)
            self.prop_phase = prop_phase

    def ar_fit_and_extrapolation(self):
        exdat, w, A, C, th  = ar_fit_interpolation(lfp= self.lfp,
                                                   PROP_LENGTH=self.PROP_LENGTH,
                                                   FS=self.FS,
                                                   AR_ORD=self.AR_ORD)
        self.exdat = exdat
        self.w     = w
        self.A     = A
        self.C     = C
        self.th    = th

        _, taper = get_w_at(self.AR_taper_k,self.WIN_SIZE) # generate another taper for the AR analysis. Note: this is not necessarily the same as the Prop taper
        self.AR_taper = taper # assign taper to class
        self.ar_phase = np.empty((int(self.WIN_SIZE),self.nChannels)) # init ar phase

        if self.nChannels > 1: # loop over channels / trials
            for j in range(self.nChannels):

                phase = ar_fourier(lfp = self.exdat[:,j],
                                            WIN_SIZE=self.WIN_SIZE,
                                            TARGET_FREQ=self.oscillation_freq,
                                            FS=self.FS,
                                            NPAD=1000,
                                            PROP_LENGTH=self.PROP_LENGTH,
                                            window_stop=self.window_stop,
                                            taper=self.AR_taper,
                                            t_before_ct=self.t_before_ct)
                self.ar_phase[:,j]   = phase

        else:
            phase = ar_fourier(lfp = np.squeeze(self.exdat),
                                        WIN_SIZE=self.WIN_SIZE,
                                        TARGET_FREQ=self.oscillation_freq,
                                        FS=self.FS,
                                        NPAD=1000,
                                        PROP_LENGTH=self.PROP_LENGTH,
                                        window_stop=self.window_stop,
                                        taper=self.AR_taper,
                                        t_before_ct=self.t_before_ct)
            self.ar_phase   = phase


    def pairwise_circ_distance(self,mode):
        """ calculates pairwise circular distance between ground truth and propagated phase """

        if mode == "prop":
            phase = self.prop_phase
        elif mode == "ar":
            phase = self.ar_phase
        if self.nChannels > 1:
            circ_distances = np.zeros((len(self.gtruth),self.nChannels))
            for jj in range(self.nChannels):
                for i in range(len(self.gtruth)):
                    circ_distances[i,jj] = np.angle(cmath.exp(1j*np.squeeze(self.gtruth[i,jj]))/cmath.exp(1j*np.squeeze(phase[i,jj]))) * np.pi/180
            
        else:
            circ_distances = np.zeros(len(self.gtruth))
            for i in range(len(self.gtruth)):
                circ_distances[i] = np.angle(cmath.exp(1j*self.gtruth[i])/cmath.exp(1j * phase[i])) * np.pi/180

        if mode == "prop":    
            self.prop_circ_distances = circ_distances
        elif mode == "ar":
            self.ar_circ_distances = circ_distances
    
    def get_stats(self):
        """ returns the mean and std of the pairwise distance """
        self.mean_distance = stats.circmean(self.distances,axis=0)
        self.circ_stds     = stats.circstd(self.distances,axis=0)
    
    def get_R(self,mode):
        if mode == "prop":
            phase = self.prop_phase
        elif mode == "ar":
            phase = self.ar_phase

        if self.nChannels > 1:
            R = np.zeros((len(self.gtruth),self.nChannels),dtype=complex)
            for jj in range(self.nChannels):
                for i in range(self.PROP_LENGTH):
                    R[i,jj] = cmath.exp(1j * (self.gtruth[i,jj] - phase[i,jj]))
        else:
            R = np.zeros((len(self.gtruth)),dtype=complex)
            for i in range(self.PROP_LENGTH):
                R[i] = cmath.exp(1j * (self.gtruth[i] - phase[i]))
        
        if mode == "prop":    
            self.prop_R = R
        elif mode == "ar":
            self.ar_R = R


    def experiment(self):
        """ runs the experiment """
        distances = np.empty((self.niter,self.PROP_LENGTH))
        Rs        = np.empty((self.niter,self.PROP_LENGTH),dtype=complex)

        for i in range(self.niter):
            self.generate_lfp() # make lfp simulation and ground truth phase data
            self.generate_gtruth() # cut the ground truth vector
            self.propagate_phase() # propagate the phase
            self.pairwise_circ_distance() # calculate the pairwise distance between ground truth and propagated phase for this iteration
            self.get_R()

            distances[i,:] = self.circ_distances # attach the pairwise distance to the experiment
            Rs[i,:] = self.R # attach the R to the experiment

        self.distances = distances # attach the distances to the experiment
        self.Rs = Rs # attach the Rs to the experiment

        # get the stats on the experiment
        self.get_stats()

    def plot_lfp(self):
        """ plots the lfp """
        fig, ax = plt.subplots(1,1,figsize=(10,5))
        ax.plot(self.lfp,label='lfp')
        ax.set_xlabel('Samples')
        ax.set_ylabel('LFP magnitude')
        ax.axvline(self.window_stop,color='w',linestyle='--',label='critical time')
        ax.set_title('LFP')
        ax.plot(np.arange(400,500),self.taper,color='r',alpha=1,label='normalized taper')
        ax.axvline(450,color='g',linestyle='--',label='taper center')
        ax.legend()
        plt.show()

    def plot_phdiffusion(self):
        """ plots the phase diffusion """
        fig, ax = plot_phase_diffusion(self.lfp,
                             self.pdata,
                             DIFFSTART=self.window_stop-self.WIN_SIZE,
                             PROP_LENGTH=self.PROP_LENGTH,
                             prop_phase=self.prop_phase)
        ax[0].axvline(self.window_stop,color='w',linestyle='--',label='critical time')
        ax[1].axvline(self.window_stop,color='w',linestyle='--',label='critical time')
        ax[0].legend()
        ax[1].legend()
        return fig, ax

    def plot_stats(self):
        """ plots the mean and std of the pairwise distance """
        fig, ax = plt.subplots(1,1,figsize=(12,5))
        ax.plot(self.mean_distance,label='sin(mean_distance)',color="orange")
        ax.fill_between(range(len(self.mean_distance)),self.mean_distance-self.circ_stds,self.mean_distance+self.circ_stds,alpha=0.6,label='circ std')
        ax.set_title(f'Mean distance from propgated to true phase with phase diffusion strength of {str(self.eps)}', color="white")

        ax.legend()
        ax.set_xlabel('Sample')
        ax.set_ylabel('Sin(distance [rad])')
        plt.show()
        return fig, ax


# %% Basic plots

sim = PhaseSimulator(
                    oscillation_freq = 10,
                    FS               = 1000,
                    WIN_SIZE         = 100,
                    window_stop      = 500,
                    PROP_LENGTH      = 100,
                    eps              = 0.1,
                    niter            = 1000,
                    random_start     = True,
                    pink_noise_level = 0.2,
                    Prop_taper_k     = -10,
                    AR_taper_k       = 0,
                    AR_ORD           = 1,
                    nChannels        = 1000,
                    nSamples         = 1000,
                    t_before_ct      = 50
                    )


sim.generate_lfp()
sim.generate_gtruth()
sim.propagate_phase()
sim.ar_fit_and_extrapolation()
sim.get_R(mode="prop")
sim.get_R(mode="ar")
sim.pairwise_circ_distance(mode="prop")
sim.pairwise_circ_distance(mode="ar")

plt.plot(abs(np.sum(sim.ar_R,axis=1)  /1000),label="ar")
plt.plot(abs(np.sum(sim.prop_R,axis=1)/1000),label ="prop")
plt.legend()


# %%

itrial = 8
fig, ax = plt.subplots(2,1,figsize=(10,9))
ax[0].plot(sim.prop_phase[:,itrial],label='propagated phase',color="orange")
ax[0].plot(sim.gtruth[:,itrial ],label='true phase')
ax[0].plot(sim.ar_phase[:,itrial ],label='ar phase',color="green")
ax[0].axvline(50,color='w',linestyle='--',label='critical time')
ax[0].legend()

ax[1].plot(np.cos(np.angle(sim.ar_R[:,itrial ])),label='ar R',color="green")
ax[1].plot(np.cos(np.angle(sim.prop_R[:,itrial ])),label='prop R',color="orange")
ax[1].axvline(50,color='w',linestyle='--',label='critical time')

ax[1].legend()
#sim.ar_fit_and_extrapolation()
#sim.get_R(mode="ar")
#sim.get_R(mode="prop")


#sim.pairwise_circ_distance(mode='ar')
#sim.pairwise_circ_distance(mode='prop')


# %%

fig, ax = plt.subplots(1,1,figsize=(8,5))
ax.hist(np.squeeze(sim.ar_circ_distances[99,:]),label='ar',alpha=0.5)
ax.hist(np.squeeze(sim.prop_circ_distances[50,:]), color='r',label='prop',alpha=0.5)
ax.legend()

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(sim.Prop_taper)
ax.plot(sim.AR_taper)






# %%
sim = PhaseSimulator(oscillation_freq = 20,
                        FS               = 1000,
                        WIN_SIZE         = 100,
                        window_stop      = 500,
                        PROP_LENGTH      = 100,
                        eps              = 0.5,
                        niter            = 1000,
                        random_start     = True,
                        pink_noise_level = 0.5,
                        Prop_taper_k     = 0,
                        AR_taper_k       = 0,
                        AR_ORD           = 50,
                        nChannels        = 500,
                        nSamples         = 1000)


sim.generate_lfp()
sim.generate_gtruth()
sim.propagate_phase()
sim.ar_fit_and_extrapolation()
sim.pairwise_circ_distance(mode='ar')
sim.pairwise_circ_distance(mode='prop')

# %%
expsamp = np.empty(shape=(2000,5000))
for j in range(5000):

    exdat, w, A, C, th  = ar_fit_interpolation( lfp          = sim.lfp[:,100],
                                                PROP_LENGTH  = 1000,
                                                FS           = 1000,
                                                AR_ORD       = 50)
    expsamp[:,j] = np.squeeze(exdat)

fig, ax = plt.subplots(3,1,figsize=(10,10))
ax[0].plot(A)
ax[1].plot(sim.lfp)
ax[2].plot(np.arange(1000,2000),np.mean(expsamp,axis=1)[1000:])


# for fitting, deal with the lfp shape. Right now its time x trials
lfp = sim.lfp
AR_ORD = 50
PROP_LENGTH=500
FS = 1000
if len(lfp.shape) > 1:
    time, ntrials = lfp.shape
    lfp    = lfp.reshape((time,1,ntrials))

else:
    time    = int(lfp.shape[0])
    lfp     = lfp[:,np.newaxis,np.newaxis] # add 2 empty axes
    ntrials = 1


# get the lfp in the right shape: time x variables(channels) x trials

v=lfp
pmin=1
pmax=AR_ORD
selector='sbc'
no_const=False
p = pmax


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






n,m,ntr = v.shape
ne      = ntr*(n-p)  # number of block equations of size m
nnp     = m*p+mcor   # number of parameter vectors of size m

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

Q, R  = np.linalg.qr(np.vstack((K,np.diag(scale))),mode='complete')









#TODO: for now we take the inpuit order as the maximum order. In the future we should include the search through the orders

# temporary
popt         = pmax
nnp          = m*popt + mcor # number of parameter vectors of length m


# decompose R for the optimal model order popt according to 
# 
#     | R11  R12 |
# R = |          |
#     | 0    R22 |
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
        w = Aaug[0,:] * con
        A = Aaug[0,1:nnp]

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




exdat = np.empty((time+PROP_LENGTH ,ntrials ))

for iTrial in range(ntrials):
    exdat[:,iTrial] = ar_interp(v=np.squeeze(lfp[:,:,iTrial]),A=A,extrasamp=PROP_LENGTH,C=C,Fs=FS)











# %% Analysis 1
eps   = 0.3
niter = 1000

def analysis_1(ks):
    """ analysis 1 """
    fig, ax = plt.subplots(len(ks),1,figsize=(10,3*len(ks)),sharex=True)

    for i,k in enumerate(ks):
        sim = PhaseSimulator(oscillation_freq=10,FS=1000,WIN_SIZE=100,window_stop=500,PROP_LENGTH=500,eps=eps,niter=niter,random_start=True,pink_noise_level=0.5,k=k)
        sim.experiment()
        ax[i].plot(abs(sum(sim.Rs)/niter),label='R')
        ax[i].plot(sim.taper/np.max(sim.taper),alpha=0.5,label='normalized taper')
        ax[i].set_title(f'Asymmetry parameter k = {str(k)}', color="white")
        ax[i].set_ylabel('R')
        if i == len(ks)-1:
            ax[i].set_xlabel('Samples')

        ax[i].axvline(sim.WIN_SIZE,color='w',linestyle='--',label='critical time')
        ax[i].legend()

    fig.suptitle(f'Phase diffusion of eps: {eps}', fontsize=20)

    plt.show()
    return fig, ax

analysis_1([-5,0,5])


# %%
from AR_tools import *


# %%
import numpy as np

sim = PhaseSimulator(oscillation_freq = 20,
                        FS               = 1000,
                        WIN_SIZE         = 100,
                        window_stop      = 500,
                        PROP_LENGTH      = 100,
                        eps              = 0.5,
                        niter            = 1000,
                        random_start     = True,
                        pink_noise_level = 0.5,
                        Prop_taper_k     = 0,
                        AR_taper_k       = 0,
                        AR_ORD           = 50,
                        nChannels        = 500,
                        nSamples         = 1000)


sim.generate_lfp()


lfp = sim.lfp
v   = lfp[:,np.newaxis,:]
p = 50
mcor = 1
n,m,ntr = v.shape
ne      = ntr*(n-p)  # number of block equations of size m
nnp     = m*p+mcor   # number of parameter vectors of size m

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
mat = np.vstack((K,np.diag(scale)))
# %%

R = scipy.linalg.qr(mat, mode='r')[0]


# %%
_, taper = get_w_at(50, 1000)

n = len(taper)
fhat = np.fft.fft(taper,n)
PSD = fhat * np.conj(fhat) / n


fig, ax = plt.subplots(2,1,figsize=(10,5))
ax[0].plot(taper)
ax[1].plot(PSD[:10])
# %%

signal = np.array([1,2,3,4,4,3,2,1])
plt.plot(signal)

plt.plot(np.fft.ifft(signal))
# %%
