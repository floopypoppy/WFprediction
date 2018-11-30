#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
analyse the error vs. the predicted frame (hidden states being reset) 

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
niter = 200

sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
sim.aoinit()
sim.makeIMat(forceNew=True)
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

'''
continuous prediction with hidden states keeping updating
'''
for k in range(n_test, n_test*2):
    sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
    sim.config.atmos.scrnNames = ['./scrns_512/scrn_%d.fits'%k]
    sim.aoinit()
    sim.makeIMat(forceNew=False)
    sim.aoloop()
    print(sim.wfsFrameNo[0])
    
    allslopes = sim.allSlopes.copy()
    wfe.extend(sim.WFE[0][1:])
    sim.aoinit()    
    _ = sim.atmos.moveScrns()
    
    for j in range(1,niter):
        inp = allslopes[:j,:].reshape((1,j,72))
        pred_slos = model.predict(inp).reshape((72,))    
        scrn = sim.atmos.moveScrns()
        
        _, phase = residualWFE(ndm,scrnsize,cmat,allslopes[j-1],iMatShapes,scrn,mask)
        phase *= phs2rad
        sr.extend([calcstrehl(sim,phase)])
        
        wfe_p, phase_p = residualWFE(ndm,scrnsize,cmat,pred_slos,iMatShapes,scrn,mask)
        wfe_pred.extend([wfe_p])
        phase_p *= phs2rad
        sr_pred.extend([calcstrehl(sim,phase_p)])
    
        wfe_b, phase_b = residualWFE(ndm,scrnsize,cmat,allslopes[j],iMatShapes,scrn,mask)
        phase_b *= phs2rad
        wfe_best.extend([wfe_b])
        sr_best.extend([calcstrehl(sim,phase_b)])

#resdic = {}
resdic = pickle.load(open('predVsframeno.p','rb'))
resdic['v=5,mag=6,ron=1'] = [wfe, wfe_pred, wfe_best, sr, sr_pred, sr_best]
pickle.dump(resdic, open('predVsframeno.p', 'wb'))

'''
plots of SR vs. frame number of delay, prediction and true
'''
sr = np.array(sr).reshape((1000,-1))
sr_pred = np.array(sr_pred).reshape((1000,-1))
sr_best = np.array(sr_best).reshape((1000,-1))
plt.plot(range(2,41), sr.mean(axis=0)[:39])
plt.plot(range(2,41), sr_pred.mean(axis=0)[:39])
plt.plot(range(2,41), sr_best.mean(axis=0)[:39])
plt.vlines(30,0.28,0.7,linestyles='dotted')
plt.xticks([2,30,40],['2','30','40'])
plt.legend(['one-frame delay','predicted zero-delay','true zero-delay'],loc='lower center')
plt.ylabel('Strehl ratio')
plt.xlabel('the predicted frame number')
plt.title('SR vs. the predicted frame number')
plt.savefig('predVsframeno.png',dpi=600)

'''
almost the same as the one above
except that both wind speed and direction are changed every *duration* AO iterations
'''
wfe = []
wfe_pred = []
wfe_best = []
sr = []
sr_pred = []
sr_best = []
vs = np.random.uniform(5,10,6)
dirs = np.random.uniform(0,2*pi,6)
niter = 60
duration = 10
n_test = 1000
looptime = sim.atmos.looptime
pxscale = sim.atmos.pixel_scale
correction = np.zeros((ndm,scrnsize,scrnsize))
for k in range(n_test):
    sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
    sim.config.atmos.scrnNames = ['./scrns_512/scrn_%d.fits'%k]
    sim.aoinit()
    sim.makeIMat(forceNew=False)
    allslopes = np.empty((niter, 72))
    
    i = 0
    windV = np.array([vs[i]*cos(dirs[i]), vs[i]*sin(dirs[i])])
    windV = windV*looptime/pxscale
    sim.atmos.windV[0] = windV               
    for j in range(niter):
        if (j+1)%duration == 0 and j!=9:
            i += 1
            windV = np.array([vs[i]*cos(dirs[i]), vs[i]*sin(dirs[i])])
            windV = windV*looptime/pxscale
            sim.atmos.windV[0] = windV           
        sim.scrns = sim.atmos.moveScrns()
        allslopes[j] = sim.runWfs(dmShape=correction, loopIter=j)
        if j == 0:
            continue
        inp = allslopes[:j,:].reshape((1,j,72))
        pred_slos = model.predict(inp).reshape((72,))    
        _, phase = residualWFE(ndm,scrnsize,cmat,allslopes[j-1],iMatShapes,sim.scrns,mask)
        phase *= phs2rad
        sr.extend([calcstrehl(sim,phase)])
        
        wfe_p, phase_p = residualWFE(ndm,scrnsize,cmat,pred_slos,iMatShapes,sim.scrns,mask)
        wfe_pred.extend([wfe_p])
        phase_p *= phs2rad
        sr_pred.extend([calcstrehl(sim,phase_p)])
        
        wfe_b, phase_b = residualWFE(ndm,scrnsize,cmat,allslopes[j],iMatShapes,sim.scrns,mask)
        phase_b *= phs2rad
        wfe_best.extend([wfe_b])
        sr_best.extend([calcstrehl(sim,phase_b)])
    
'''
plots of SR vs. frame number of delay, prediction and true with changing wind velocities, with arrows showing wind vectors if needed
'''
n_test = 500    
niter = 60
sr = np.array(sr).reshape((n_test,-1))
sr_pred = np.array(sr_pred).reshape((n_test,-1))
sr_best = np.array(sr_best).reshape((n_test,-1))
windV = np.array([vs*cos(dirs), vs*sin(dirs)])
plt.plot(range(2,niter+1), sr.mean(axis=0))
plt.plot(range(2,niter+1), sr_pred.mean(axis=0))
plt.plot(range(2,niter+1), sr_best.mean(axis=0))
plt.xticks([2,20,30,40,50,60])#,['2','20','30','40','50','60'])
#plt.vlines([20,30,40,50],0.4,0.75,linestyles='dotted')
#plt.text(5,0.38,'(10,pi)')
#plt.text(19,0.36,'(10,0)')
#plt.text(29,0.38,'(5,pi)')
#plt.text(39,0.36,'(5,0)')
#plt.text(49,0.38,'(10,1.5pi)')
#quiver([10,25,35,45,55],0.43,windV[0][:-1],windV[1][:-1],width=0.003)
plt.legend(['delay','prediction','true'])
plt.xlabel('frame number')
plt.ylabel('Strehl ratio')
plt.title('SR vs frame number (v=5-10m/s, theta=0)')
plt.savefig('SRvsFrameNo(changingV-speedGradual.png',dpi=600,overwrite=True)

'''
plots of bandwidth error vs. frame number
'''
alldic = pickle.load(open('/users/xliu/dropbox/expout/DoubleLys.p','rb'))
resdic = alldic['James12Profile_atm1_halfspeed']
n_test = 500
niter = 40
wfe = resdic[0]
wfe_pred = resdic[1]
wfe_best = resdic[2]
rsmd_pred = [rsmd(wfe_best[:,i], wfe_pred[:,i]) for i in range(niter-1)]
rsmd_delay = [rsmd(wfe_best[:,i], wfe[:,i]) for i in range(niter-1)]
plt.plot(range(2,niter+1), rsmd_delay)
plt.plot(range(2,niter+1), rsmd_pred)
plt.xticks([2,10,20,30,40])
plt.yticks([40,60,600])
plt.legend(['delay','prediction'])
plt.xlabel('frame number')
plt.ylabel('Bandwidth error (nm)')
plt.title('BW error vs frame number: 4-layer profile (half speed')
plt.savefig('BWErrvs.Frame_4lys_halfspeed.png',dpi=600,overwrite=True)