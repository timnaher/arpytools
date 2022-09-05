# %%
import numpy as np
from scipy.fft import fft, fftshift
import scipy.signal as sig
from math import *
import cmath
from fooof import FOOOF
from scipy.signal import savgol_filter
import colorednoise as cn
import matplotlib.pyplot as plt
from sim_tools import *
from plotting_tools import *

plt.style.use("dark_background")

# %%
oscillation_freq = 10
FS = 1000
WIN_SIZE = 100
window_stop = 500
PROP_LENGTH = 500

# create lfp data and groundtruth phase
lfp, pdata = sim_lfp(
    oscillation_freq, nChannels=1, eps=0.1, FS=FS, nSamples=1000, pink_noise_level=0.5
)


# this function propagates the phase based on a selected fourier coefficient of the lfp
prop_phase = propagate_phase(
    lfp,
    WIN_SIZE=WIN_SIZE,
    TARGET_FREQ=oscillation_freq,
    FS=FS,
    NPAD=1000,
    PROP_LENGTH=PROP_LENGTH,
    window_stop=window_stop,
)
# plot both signal and phase
# plot_phase_diffusion(lfp,pdata,DIFFSTART=window_stop-WIN_SIZE,PROP_LENGTH=PROP_LENGTH,prop_phase=prop_phase)


# sample to sample distance between groundtruth and propagated phase
gtruth = pdata[window_stop - WIN_SIZE : (window_stop - WIN_SIZE + PROP_LENGTH)]

circ_distances = np.zeros(len(gtruth))
for i in range(len(gtruth)):
    circ_distances[i] = (
        np.angle(cmath.exp(1j * gtruth[i]) / cmath.exp(1j * prop_phase[i]))
        * np.pi
        / 180
    )

plot_phase_diffusion(
    lfp,
    pdata,
    DIFFSTART=window_stop - WIN_SIZE,
    PROP_LENGTH=PROP_LENGTH,
    prop_phase=prop_phase,
)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(circ_distances, label="circ_distances", color="orange")
ax[1].plot(gtruth, label="gtruth", color="orange")
ax[1].plot(prop_phase, label="prop_phase", color="red")
ax[0].set_title("circ_distances", color="white")
ax[1].set_title("gtruth", color="white")
ax[1].set_title("prop_phase", color="white")

# plt.plot(circ_distances,label='circ_distances')

# %%
