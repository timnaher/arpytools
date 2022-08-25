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
                 AR_taper_k            = 0,    # asymmetry of taper. posivie = right skewerd, negative = left skewed, 0 = no skew              
                 nSamples              = 1000, # number of samples in LFP data
                 nChannels             = 1,   # number of LFP channels or alternatively LFP trials
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
        self.gtruth     = self.pdata[self.window_stop-self.WIN_SIZE : (self.window_stop-self.WIN_SIZE+self.PROP_LENGTH)]

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
                                    taper=self.Prop_taper)
                self.prop_phase[:,j]   = prop_phase
        else:
            prop_phase = propagate_phase(self.lfp,WIN_SIZE=self.WIN_SIZE,
                                    TARGET_FREQ=self.oscillation_freq,
                                    FS=self.FS,
                                    NPAD=1000,
                                    PROP_LENGTH=self.PROP_LENGTH,
                                    window_stop=self.window_stop,
                                    taper=self.Prop_taper)
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
                                            taper=self.AR_taper)
                self.ar_phase[:,j]   = phase

        else:
            phase = ar_fourier(lfp = np.squeeze(self.exdat),
                                        WIN_SIZE=self.WIN_SIZE,
                                        TARGET_FREQ=self.oscillation_freq,
                                        FS=self.FS,
                                        NPAD=1000,
                                        PROP_LENGTH=self.PROP_LENGTH,
                                        window_stop=self.window_stop,
                                        taper=self.AR_taper)
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
    
    def get_R(self):
        R = np.empty(self.PROP_LENGTH,dtype=complex)
        for i in range(self.PROP_LENGTH):
            R[i] = cmath.exp(1j * (self.gtruth[i] - self.prop_phase[i]))
            self.R = R
    
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
