import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt
import scipy.signal as sig
import random
import cmath


def phase_diffusion(
    freq, eps=0.1, FS=1000, nChannels=2, nSamples=1000, random_start=True
):

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

    if random_start:  # if random start, add a random phase
        phases += random.uniform(0, 2 * np.pi)

    return phases % (2 * np.pi)


def sim_lfp(
    osc_freq,
    nChannels=1,
    eps=0.1,
    FS=1000,
    nSamples=1000,
    pink_noise_level=1,
    random_start=True,
):
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

    pdata = phase_diffusion(
        freq=osc_freq,
        nChannels=nChannels,
        eps=eps,
        FS=FS,
        nSamples=nSamples,
        random_start=random_start,
    )
    noise = (
        np.squeeze(cn.powerlaw_psd_gaussian(1, (nChannels, 1, nSamples)))
        * pink_noise_level
    )
    lfp = np.cos(pdata[:, 0]) + noise

    return lfp, pdata


def propagate_phase(
    lfp, WIN_SIZE, TARGET_FREQ, FS, NPAD, PROP_LENGTH, window_stop, taper
):
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

    # hann tapered fft to get FCs
    snippet = lfp[(window_stop - WIN_SIZE) : window_stop] * taper
    freqs = np.fft.rfftfreq(NPAD, 1 / FS)
    frq_indx = np.where(freqs == TARGET_FREQ)[0]
    spec = np.fft.rfft(snippet, n=NPAD)
    ph = np.angle(spec)[frq_indx]
    prop_ph = (ph + TARGET_FREQ * 2 * np.pi / FS * np.arange(PROP_LENGTH)) % (
        2 * np.pi
    )  # modulo 2pi, propagated phase

    return prop_ph


def get_w_at(k, N):
    # make hann taper
    w_s = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    # get hann phase
    w_s_phase = np.angle(sig.hilbert(w_s))

    w_at = np.empty(N)
    for n in range(N):
        w_at[n] = w_s[n] * cmath.exp(k * w_s_phase[n])

    c = N / (2 * np.sum(w_at))  # normalization factor
    return w_s, w_at * c  # normalize and return
