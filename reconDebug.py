#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
response of each DM (computed DM command) to each influence function as input phase

"""

import scipy

sim = soapy.Sim("sh_77_openloop_piezo_150Hz.yaml")
sim.aoinit()
sim.makeIMat(forceNew=True)
imat = sim.recon.interaction_matrix

n_acts = sim.dms[0].n_acts
dmCmds = np.zeros((n_acts, n_acts))
imatshapes = sim.dms[0].iMatShapes
slopes = np.zeros((n_acts, 72))

for i in range(n_acts):
    slopes[i] = sim.wfss[0].frame(imatshapes[i])

for svdCond in [0.1]:#np.arange(0.08, 0.21, 0.01):
    cmat = scipy.linalg.pinv(imat, svdCond)
    for i in range(n_acts):
        dmCmds[i] = cmat.T.dot(slopes[i])
    print(np.trace(dmCmds))
    
imshow(dmCmds, extent=[0.5,n_acts+0.5,n_acts+0.5,0.5])
plt.colorbar()
plt.xticks([1,n_acts])
plt.yticks([1,n_acts])
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False)
plt.savefig('zernike_response_{}_800.png'.format(n_acts), dpi=300)

imshow(wfs.detector)

_, s, _ = svd(imat)
plt.plot(s)

var = []
for i in range(n_acts):
#    print(imatshapes[i].flatten().max())
    var.append(np.sqrt(imatshapes[i].flatten().var()))

"""compare correction by Piezo DM and by Zernike DM"""
n_run = 10
instStrehl = []
sim = soapy.Sim("sh_77_openloop_zernike.yaml")
sim.aoinit()
sim.makeIMat(forceNew=True)
for i in range(n_run):
    sim.aoinit()
    sim.makeIMat()
    sim.aoloop()
    instStrehl.append(sim.instStrehl[0][5:].mean())
    

sim.scrns = (imatshapes[30]*2).reshape((1,66,66))
_ = sim.runWfs()
imshow(sim.wfss[0].detector)