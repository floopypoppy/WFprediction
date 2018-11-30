#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Run a continuous AO open-loop simulation and gather slopes from each time step. Compare WFE/SR of the LAST frame ONLY with one-frame delay, prediction or zero-delay.

"""
def residualWFE(ndm,scrnsize,cmat,slos,iMatShapes,scrn,mask):
    '''
    part of Soapy code
    combine current atmos phase screen and slope fed to DM to compute corrected phase and residual wavefront error 
    '''
    correction = np.zeros((ndm, scrnsize, scrnsize))
    dmCommands = cmat.T.dot(slos)

    dmshape = (iMatShapes.T*dmCommands[:].T).T.sum(0)
    dmshape -= dmshape.mean()
    
    j = 0
    for DMshape in [dmshape]:
        dmsize = DMshape.shape[0]
        if dmsize == scrnsize:
            correction[j] = DMshape.copy()
        else:
            if dmsize > scrnsize:
                coord = int(round(dmsize/2. - scrnsize/2.))
                correction[j] = DMshape[coord: -coord, coord: -coord]
    
            else:
                pad = int(round((scrnsize - dmsize)/2))
                correction[j, pad:-pad, pad:-pad] = DMshape      
        j += 1
    
    phase = scrn.copy().sum(axis=0)
    correction = correction.sum(0)
    phase -= correction
    res = phase.copy() * mask
    piston = res.sum() / mask.sum()
    res -= (piston*mask)
    ms_wfe = np.square(res).sum() / mask.sum()
    rms_wfe = np.sqrt(ms_wfe)
    return rms_wfe, phase

def calcstrehl(sim, phase):
    '''
    part of Soapy code
    compute Strehl ratio of science camera from corrected phase  
    '''
    sim.sciCams[0].los.phase = phase.copy()
    sim.sciCams[0].calcFocalPlane()
    sim.sciCams[0].calcInstStrehl()
    return sim.sciCams[0].instStrehl
    
def lmean(alist):
    '''calculate mean value of a list'''
    return np.array(alist).mean()

def lstd(alist):
    '''calculate standard deviation of a list'''
    return np.array(alist).std()

file_name = '1ly_72_72_20_128_30_withTT(lr=0.001)_gauss_0.1'
model = load_model('/users/xliu/dropbox/expout/'+file_name+'.h5')
n_test = 1000
ndm = 1
scrnsize = 130
niter = 30

sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
sim.aoinit()
sim.makeIMat(forceNew=False)
iMatShapes = sim.dms[0].iMatShapes
cmat = sim.recon.control_matrix
mask = sim.mask
phs2rad = sim.sciCams[0].los.phs2Rad

wfe = []
wfe_pred = []
wfe_best = []
sr = []
sr_pred = []
sr_best = []


for k in range(niter):
    sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
    sim.config.atmos.scrnNames = ['./scrns_512_2/scrn_%d.fits'%k]
    sim.aoinit()
    sim.makeIMat(forceNew=False)
    sim.aoloop()
    wfe.append(sim.WFE[0][-1])
    sr.append(sim.sciCams[0].instStrehl)
    
    slopes = sim.allSlopes
    
    inp = slopes[0:-1,:].reshape((1,niter-1,72))
    pred_slos = model.predict(inp).reshape((72,))    
    scrn = sim.atmos.scrns.copy()
    
    wfe_p, phase_p = residualWFE(ndm,scrnsize,cmat,pred_slos,iMatShapes,scrn,mask)
    wfe_pred.append(wfe_p)
    phase_p *= phs2rad
    sr_pred.append(calcstrehl(sim,phase_p))

    wfe_b, phase_b = residualWFE(ndm,scrnsize,cmat,slopes[-1],iMatShapes,scrn,mask)
    phase_b *= phs2rad
    wfe_best.append(wfe_b)
    sr_best.append(calcstrehl(sim,phase_b))

resdic = {}
resdic['v=5,mag=6,ron=1'] = [wfe, wfe_pred, wfe_best, sr, sr_pred, sr_best]
pickle.dump(resdic, open('SRvsSpeed.p', 'wb'))