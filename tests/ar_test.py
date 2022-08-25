# %%
import numpy as np
from AR_tools import *
from phase_statistics import PhaseSimulator

# test the AR function on the simulated data

# %%
sim = PhaseSimulator(pink_noise_level=0)
sim.generate_lfp()
sim.generate_gtruth_and_pphase()
plt.plot(sim.lfp)

# copy the lfp over to  new var
lfp = np.copy(sim.lfp)

# train AR model on the data
v    = np.vstack((lfp))
v    = v.reshape((v.shape[0],1,1))

# %%
w, A, C, th = arfit(v,50,50,selector='sbc',no_const=False)

Fs        = 1000
extrasamp = 1000

exdat     = ar_interp(v,A,extrasamp,Fs)

fig, ax = plt.subplots(3,1,figsize = (8,10),constrained_layout=True)
fig.tight_layout()
plt.style.use('dark_background')
ax[0].plot(v[:,0,0])
#ax[0].plot(v[:,0,2])

ax[0].set_title('Original Signal')
ax[1].plot(A[0,:])
ax[1].set_title('AR Coefficients')
ax[2].plot(np.arange(2000),exdat)
#ax[2].plot(np.arange(200,200+extrams),exdat[origsamps:])
ax[2].set_title('Extrapolated Signal')
ax[2].set_xlabel('Time (ms)')
ax[2].legend(['Original','Extrapolated'])
plt.show()

# %%
