# %%
from PhaseSimulator import *



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
                        nChannels        = 1,
                        nSamples         = 1000)


sim.generate_lfp()
sim.generate_gtruth()
sim.propagate_phase()
sim.ar_fit_and_extrapolation()
sim.pairwise_circ_distance(mode='ar')
sim.pairwise_circ_distance(mode='prop')
# %%

expsamp = np.empty(shape=(1000,1000))

for j in range(1000):

    exdat, w, A, C, th  = ar_fit_interpolation( lfp          = sim.lfp,
                                                PROP_LENGTH  = 1000,
                                                FS           = 1000,
                                                AR_ORD       = 50)
    expsamp[:,j] = np.squeeze(exdat[1000:])

plt.plot(np.mean(expsamp,axis=1))
plt.plot(A)
# %%
