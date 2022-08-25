import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt
import scipy.signal as sig
import random
import cmath

from AR_tools import *
import tqdm
from numba import njit, prange




def phase_diffusion(freq,
                    eps=.1,
                    FS=1000,
                    nChannels=2,
                    nSamples=1000,
                    random_start=True):

    """
    Linear (harmonic) phase evolution + a Brownian noise term
    inducing phase diffusion around the deterministic phase drift with
    slope ``2pi * freq`` (angular frequency).

    The linear phase increments are given by ``dPhase = 2pi * freq/fs``,
    the Brownian increments are scaled with `eps` relative to these
    phase increments.

    Parameters
    ----------
    freq : float
        Harmonic frequency in Hz
    eps : float
        Scaled Brownian increments
        `1` means the single Wiener step
        has on average the size of the
        harmonic increments, so very strong
        diffusion
    fs : float
        Sampling rate in Hz
    nChannels : int
        Number of channels
    nSamples : int
        Number of samples in time
    return_phase : bool, optional
        If set to true returns the phases in radians

    Returns
    -------
    phases : numpy.ndarray
        Synthetic `nSamples` x `nChannels` data array simulating noisy phase
        evolution/diffusion
    """

    # white noise
    wn = np.random.randn(nSamples, nChannels)
    delta_ts = np.ones(nSamples) * 1 / FS
    omega0 = 2 * np.pi * freq
    lin_incr = np.tile(omega0 * delta_ts, (nChannels, 1)).T

    # relative Brownian increments
    rel_eps = np.sqrt(omega0 / FS * eps)
    brown_incr = rel_eps * wn

    # add both together
    phases = np.cumsum(lin_incr + brown_incr, axis=0)

    if random_start: # if random start, add a random phase
        phases = ((phases % (2 * np.pi)) + random.uniform(0, 2*np.pi)) % (2*np.pi)
    else:
        phases = phases % (2 * np.pi)

    return phases 

def sim_lfp(osc_freq,nChannels=1,eps=0.1,FS=1000,nSamples=1000,pink_noise_level=1,random_start=True):
    """simulates an lfp singla with a known noisey phase diffusion oscillator and some 1/f noise

    Args:
        osc_freq (int): oscillator frequency in Hz
        nChannels (int, optional): number of channels. Defaults to 1.
        eps (float, optional): _description_. Defaults to 0.1.
        FS (int, optional): sampling frequency. Defaults to 1000.
        nSamples (int, optional): length of samples. Defaults to 1000.
        pink_noise_level (int, optional): proportion of noise level. Defaults to 1.

    Returns:
        np.array: chan x time array of simulated lfp
    """    
    
    pdata  = phase_diffusion(freq=osc_freq, nChannels=nChannels,eps=eps,FS=FS,nSamples=nSamples,random_start=random_start)
    noise  = np.squeeze(cn.powerlaw_psd_gaussian(1,(nChannels, 1, nSamples))) * pink_noise_level
    lfp    = np.cos(pdata[:,0]) + noise

    return lfp, pdata

def propagate_phase(lfp,WIN_SIZE,TARGET_FREQ,FS,NPAD,PROP_LENGTH,window_stop,t_before_ct,taper):
    """propagates the phase based on a selected fourier coefficient of the lfp

    Args:
        lfp (array): array with signal
        WIN_SIZE (int): hann taper window size (symmetric)
        TARGET_FREQ (int): frequency of fourier coefficient to propagate
        FS (int): sampling rat ein Hz
        NPAD (int): to where to pad the signal before fft 
        PROP_LENGTH (int): length of the propagation
        window_start_stop (list): start and stop of the window to propagate

    Returns:
        array: propagated phase from beginning of window to selected point
    """

    # tapered fft to get FCs
    # multiply snippet by taper
    snippet    = lfp[ int(window_stop-WIN_SIZE):window_stop] * taper
    freqs      = np.fft.rfftfreq(NPAD, 1 / FS)
    frq_indx   = np.where(freqs==TARGET_FREQ)[0]
    spec       = np.fft.rfft(snippet,n=NPAD)
    ph         = np.angle(spec)[frq_indx]

    # get the length to propagate the phase on
    plength = len(taper) + PROP_LENGTH - t_before_ct # this propagates from the start of the taper on
    phase = (ph + TARGET_FREQ * 2 * np.pi / FS * np.arange(plength)) % (2 * np.pi)
    return phase[t_before_ct:]


def ar_fourier(lfp,WIN_SIZE,TARGET_FREQ,FS,NPAD,PROP_LENGTH,window_stop,t_before_ct,taper):
    """propagates the phase based on a selected fourier coefficient of the lfp
    NOTE:
    
    
    """
    # tapered fft to get FCs on extrp data
    snippet    = lfp[ int(window_stop-WIN_SIZE/2) : int(window_stop+WIN_SIZE/2)] * taper
    freqs      = np.fft.rfftfreq(NPAD, 1 / FS)
    frq_indx   = np.where(freqs==TARGET_FREQ)[0]
    spec       = np.fft.rfft(snippet,n=NPAD)
    # ph reflects the phase of estimate at the beginning of the taper, therefore the phase
    # at window_stop-WIN_SIZE/2. In order to get the phase at the critical time, we need to
    # propagate it forward by the length of the taper window divided by 2
    ph         = np.angle(spec)[frq_indx]
    # get the length to propagate the phase on
    plength = len(taper) + PROP_LENGTH - t_before_ct # this propagates from the start of the taper on
    phase = (ph + TARGET_FREQ * 2 * np.pi / FS * np.arange(plength)) % (2 * np.pi)
    return phase[t_before_ct:]

def ar_fit_interpolation(lfp,PROP_LENGTH,FS,AR_ORD):
    # for fitting, deal with the lfp shape. Right now its time x trials
    if len(lfp.shape) > 1:
        time, ntrials = lfp.shape
        lfp    = lfp.reshape((time,1,ntrials))

    else:
        time    = int(lfp.shape[0])
        lfp     = lfp[:,np.newaxis,np.newaxis] # add 2 empty axes
        ntrials = 1


    # get the lfp in the right shape: time x variables(channels) x trials
    w, A, C, th = arfit(v=lfp,pmin=1,pmax=AR_ORD,selector='sbc',no_const=False)
    exdat = np.empty((time+PROP_LENGTH ,ntrials ))

    for iTrial in range(ntrials):
        exdat[:,iTrial] = ar_extrap(v=np.squeeze(lfp[:,:,iTrial]),A=A,extrasamp=PROP_LENGTH,C=C,Fs=FS)

    return exdat, w, A, C, th


def get_w_at(k,N):
    # make hann taper
    w_s = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    # get hann phase
    w_s_phase = np.angle(sig.hilbert(w_s)) 
    
    w_at      = np.empty(N)
    for n in range(N):
        w_at[n] =   w_s[n] * cmath.exp(k * w_s_phase[n]) 

    c = N / (2 * np.sum(w_at)) # normalization factor
    return w_s, w_at*c # normalize and return