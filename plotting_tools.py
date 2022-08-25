import matplotlib.pyplot as plt
import numpy as np

def plot_phase_diffusion(lfp,pdata,DIFFSTART=None,PROP_LENGTH=None,prop_phase=None):
    # plot both
    fig, ax = plt.subplots(2,1,figsize=(10,10))

    # plot scaled signals
    ax[0].plot(lfp * 3 + 3, label='Pseudo LFP signal')
    ax[0].set_title('signal',color="white")
    ax[1].plot(pdata, label='phase',color="orange")
    ax[1].set_title('Ground truth phase',color="white")

    # Optional: plot the propagated phase
    ax[1].plot(np.arange(DIFFSTART,DIFFSTART + PROP_LENGTH),prop_phase, label='propagated phase',color="red")

