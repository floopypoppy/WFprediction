#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Two approaches for calculating bandwidth error.
----Approach one----
In the simulation, ron is first turned off and then turned on. Delay is compensated by using the true slopes for correction. 
Results from the first round can be used for computing fitting errors + other errors. It should be smaller than theoretical fitting errors only. 
Results from the second round combined with the first round's can be used for computing ron-related error term. It is then compared with theoretical expressions to check its validity.
Removing fitting+other error and RON error, we get BW error.
----Approach two----
Compare best WFE (contains fitting error, other error and RON error only) and delay/predicted WFE (contains additional bandwidth error) to get BW errors.
Finally is BW error with respect to r0, GS magnitude wind speeds and etc. calculated from Approach two.
'''

def residualWFE(ndm,scrnsize,cmat,slos,iMatShapes,scrn,mask):
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
    sim.sciCams[0].los.phase = phase.copy()
    sim.sciCams[0].calcFocalPlane()
    sim.sciCams[0].calcInstStrehl()
    return sim.sciCams[0].instStrehl
    
def lmean(alist):
    return np.array(alist).mean()

def lstd(alist):
    return np.array(alist).std()

n_test = 1000
ndm = 1
scrnsize = 130

sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
sim.aoinit()
sim.makeIMat(forceNew=True)
iMatShapes = sim.dms[0].iMatShapes
cmat = sim.recon.control_matrix

mask = sim.mask
phs2rad = sim.sciCams[0].los.phs2Rad
wfe_best = []
sr_best = []


for k in range(n_test):
    sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
    sim.aoinit()
    sim.makeIMat(forceNew=False)
    sim.aoloop()
    
    slopes = sim.allSlopes
  
    scrn = sim.atmos.scrns.copy()

    wfe_b, phase_b = residualWFE(ndm,scrnsize,cmat,slopes[-1],iMatShapes,scrn,mask)
    phase_b *= phs2rad
    wfe_best.append(wfe_b)
    sr_best.append(calcstrehl(sim,phase_b))

resdic = pickle.load(open('otherNoise.p','rb'))
otherNoise = lmean(resdic['v=5'])


'''
compute theoretical fitting errors for sanity check
'''

def fitErr(d, r0, lmda, af):
    '''
    compute theoretical fitting error in nm
    '''
    import numpy as np
    r = (lmda/500e-9)**(6/5.)*r0
    alphaf_squared = af*(d/r)**(5/3.)
    wfe = np.sqrt(alphaf_squared)/np.pi/2.*lmda*1e9
    return wfe

'''
compute theoretical RON-related WFE in nm
'''

def noiseErr(n_ph, n_r, n_s, n_samp, lmda):
    '''
    compute RON-related phase error (nm)
    from S. Thomas 2005 paper Eq. (12)
    used when the centroid algorithm is CoG
    '''
    import numpy as np 
    numerator = np.pi**2 * n_r**2 * n_s**4
    denominator = 3 * n_ph**2 * n_samp**2
    alphan_squared = numerator/denominator
    wfe = np.sqrt(alphan_squared)/np.pi/2.*lmda*1e9
    return wfe

def calNsamp(lmda, d, p):
    return lmda/d/p

def noiseErr_new(d, s, z, ron, F):
    '''
    compute RON related OPD RMS between opposite edges of a subap (nm)
    MOSAIC report 'E-MOS-TEC-ANR-0015'
    original paper: S. Thomas 2008 Eq. (6)
    used when the centroid algorithm is WCoG
    '''
    import numpy as np
    numerator = 0.82 * d**2 * ron**2 * s**4
    denominator = z**2 * F**2
    alphan_squared = numerator/denominator
    wfe = np.sqrt(alphan_squared)*1e9
    return wfe

'''set relevant parameters'''
d = 4.2/7
lmda = 600e-9
r0 = 0.16
r = r0*(lmda/500e-9)**(6/5.)
s = lmda/r
z = 4.8/16*arc2rad
ron = 1
mag = 10
from soapy.wfs.shackhartmann import photons_per_mag
sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
sim.aoinit()
wfs = sim.wfss[0]
mask = wfs.mask
phase_scale = 4.2/128
exposureTime = 0.012
zeropoint = 2e9
F = photons_per_mag(mag, mask, phase_scale, exposureTime, zeropoint)/36
n_samp = calNsamp(lmda, d, z)

from soapy.wfs.shackhartmann import photons_per_mag
sim = soapy.Sim("../soapy-master/conf/sh_77_openloop.yaml")
sim.aoinit()
#mag = 10
wfs = sim.wfss[0]
mask = wfs.mask
phase_scale = 4.2/128
exposureTime = 0.012
zeropoint = 2e9
#n_ph = photons_per_mag(mag, mask, phase_scale, exposureTime, zeropoint)/36
d = 4.2/7
p = 4.8/16*arc2rad
r0 = 0.16
n_s = 16
lmda = 600e-9 # won't affect calculated WFE in nm
n_samp = calNsamp(lmda, d, p)
af = 0.3
n_r = 1

#ronerr = noiseErr(n_ph, n_r, n_s, n_samp, lmda)

'''
compare theoretical formulae and simulation results to verify RON-related WFE formulae
'''
def rmsd(true, sample):
    '''
    root mean squared diff
    return root mean of (sample^2 - true^2)
    '''
    import numpy as np
    diff = sample**2 - true**2
    index = np.where(diff >= 0)
    return np.sqrt(diff[index].mean())

def rsmd(true, sample):
    '''
    return root of (sample_mean^2 - true_mean^2)
    '''
    true_m = np.array(true).mean()
    true_s = np.array(sample).mean()
    try:
        return np.sqrt(true_s**2 - true_m**2)
    except:
        return 0
    
def new(true, sample):
    '''
    return root of (sample^2_mean - true^2_mean)
    '''
    true_m = np.array(true)**2
    true_s = np.array(sample)**2
    try:
        return np.sqrt(true_s.mean() - true_m.mean())
    except:
        return 0
    

'''simulation results'''
bwwfe_real = []
alldic = pickle.load(open('/users/xliu/dropbox/expout/RONoise.p','rb'))
for i in [6,8,9,10]:
    wfe = alldic['v=10,mag=%d,ron=0'%i]
    wfe_n = alldic['v=10,mag=%d,ron=1'%i]
    bwwfe_real.append(rsmd(wfe, wfe_n))
    
'''computed theoretical results - CoG formula'''
bwwfe_ideal = []
for m in [6,8,9,10]:
    mag = m
    n_ph = photons_per_mag(mag, mask, phase_scale, exposureTime, zeropoint)/36
    bwwfe_ideal.append(noiseErr(n_ph, n_r, n_s, n_samp, lmda))
    
plt.semilogy([6,8,9,10],bwwfe_real)
plt.semilogy([6,8,9,10],bwwfe_ideal)
plt.xticks([6,8,9,10],['6','8','9','10'])
plt.legend(['simulated','theoretical'])
plt.xlabel('GS magnitude')
plt.ylabel('RON related WFE (nm)')
plt.title('Theoretical and simulated RON related WFE vs. GS magnitude (RON=1e-)')
plt.savefig('RON-WFEvs.GSmag_2.png',dpi=600)

'''computed theoretical results two - WCoG formula'''
bwwfe_ideal2 = []
for m in [6,8,9,10]:
    mag = m
    F = photons_per_mag(mag, mask, phase_scale, exposureTime, zeropoint)/36
    bwwfe_ideal2.append(noiseErr_new(d, s, z, ron, F))

plt.semilogy([6,8,9,10],bwwfe_real)
plt.semilogy([6,8,9,10],bwwfe_ideal)
plt.semilogy([6,8,9,10],bwwfe_ideal2)
plt.xticks([6,8,9,10],['6','8','9','10'])
plt.legend(['simulated','theoretical-Sandrian paper','theoretical-MOSAIC domument'])
plt.xlabel('GS magnitude')
plt.ylabel('RON related WFE (nm)')
plt.title('Theoretical and simulated RON related WFE vs. GS magnitude (RON=1e-)')

'''
another way to approach this problem
compare best WFE (contains fitting error, other error and RON error only) and delay/predicted WFE (contains additional bandwidth error) to get BW errors
analyse BW error with respect to r0, GS magnitude wind speeds and etc. 
'''
bwwfe_simu = []
alldic_simu = pickle.load(open('/users/xliu/dropbox/expout/RONoise_gaussian.p','rb'))
for i in [6,8,9,10]:
    resdic = alldic_simu['v=10,mag=%d,ron=1,svdcond=0.15,sd=0.1'%i]
    wfe = np.array(resdic[0])
    wfe_pred = np.array(resdic[1])
    wfe_best = np.array(resdic[2])   
    bwwfe_simu.append(new(wfe_best, wfe))
    bwwfe_simu.append(new(wfe_best, wfe_pred))   

plt.plot([6,8,9,10], np.array(bwwfe_simu).reshape((-1,2)))
plt.legend(['delay','prediction'])
plt.xlabel('GS magnitude')
plt.ylabel('bandwidth WFE (nm)')
plt.xticks([6,8,9,10],['6','8','9','10'])
plt.title('Bandwidth error (nm) vs. GS magnitude')
plt.savefig('BWErrvs.GSmag.png', dpi=600)
    
'''simulation two'''
bwwfe_simu = []
alldic_simu = pickle.load(open('SRvsSpeed.p','rb'))
for i in range(5,17):
    resdic = alldic_simu['v=%d,mag=6,ron=1'%i]
    wfe = np.array(resdic[0])
    wfe_pred = np.array(resdic[1])
    wfe_best = np.array(resdic[2])   
    bwwfe_simu.append(rsmd(wfe_best, wfe))
    bwwfe_simu.append(rsmd(wfe_best, wfe_pred))   
    
plt.plot(range(5,17), np.array(bwwfe_simu).reshape((-1,2)))
plt.legend(['delay','prediction'])
plt.xlabel('wind speed (m/s)')
plt.ylabel('bandwidth WFE (nm)')
plt.title('Bandwidth error (nm) vs. test wind speed (direction=0)')
plt.savefig('BWErrvs.Speed.png', dpi=600)

'''simulation three'''
bwwfe_simu = []
alldic = pickle.load(open('/users/xliu/dropbox/expout/SRvsR0.p','rb'))
for r in np.arange(0.04,0.25,0.04):
    resdic = alldic['v=10,mag=6,ron=1,r0=%.2f'%r]
    wfe = resdic[0]
    wfe_pred = resdic[1]
    wfe_best = resdic[2]
    bwwfe_simu.append(rsmd(wfe_best, wfe))
    bwwfe_simu.append(rsmd(wfe_best, wfe_pred))

plt.semilogy(np.log10(range(4,25,4)), np.array(bwwfe_simu).reshape((-1,2)))
plt.legend(['delay','prediction'])
plt.xlabel('r0 (cm)')
plt.ylabel('bandwidth WFE (nm)')
plt.title('Bandwidth error (nm) vs. r0')
plt.xticks(np.log10(range(4,25,4)),['4','8','12','16','20','24'])
plt.yticks(np.array([60,600]),['60','600'])
plt.savefig('BWErrvs.R0.png', dpi=600)

    