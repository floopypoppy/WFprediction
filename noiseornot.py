#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ANN test:
    ANN performance (WFE) when guide star magnitude changes

"""
#%%
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

def dmCmdDiff_to_WFE(dmCommands,iMatShapes,mask):
    '''
    conbine input dmcommands difference (converted from two different sets of wavefront measurements via control matrix) and dm influence function to compute phase difference map, then output its mean squared value over pupil mask
    '''

    dmshape = (iMatShapes.T*dmCommands[:].T).T.sum(0)
    dmshape -= dmshape.mean() 
    res = dmshape.copy() * mask
    piston = res.sum() / mask.sum()
    res -= (piston*mask) # mean of all non-zero value (aperture part) of res is 0 -- piston removed
    ms_wfe = np.square(res).sum() / mask.sum()
    rms_wfe = np.sqrt(ms_wfe)
    return rms_wfe

ntest = 1000
seqlen = 100

#%%
#file_name = '1ly_235_242_0_0_40_128_30_withTT_(lr=0.0005)_dp=0-0.1-0-0.1_0402_2.h5'
#model = load_model('/users/xliu/dropbox/expout/'+file_name)

mag = 7
#cleanslos = fits.getdata('/users/xliu/dropbox/expout/slopes_0.0067_v=15_noisefree_mag=%d.fits'%mag).reshape((ntest,-1))
noisyslos = fits.getdata('/users/xliu/dropbox/expout/slopes_0.0067_v=15_noisefree_mag=%d.fits'%mag).reshape((ntest,-1))

delayslos = noisyslos[:,:(seqlen-1)*72].reshape((-1,72))
trueslos = noisyslos[:,72:].reshape((-1,72))
#truth = cleanslos[:,72:].reshape((-1,72))
#truth = cleanslos[:(seqlen-1)*72].reshape((-1,72))

#fits.writeto('/users/xliu/Dropbox/expout/predslos_0.0067_v=15_mag=8.fits',predslos)

#fits.writeto('/users/xliu/Dropbox/expout/delayslos_0.0067_v=10_mag=7.fits',delayslos)
#fits.writeto('/users/xliu/Dropbox/expout/trueslos_0.0067_v=10_mag=7.fits',trueslos)
#fits.writeto('/users/xliu/Dropbox/expout/truth_0.0067_v=15_mag=%d.fits'%mag, delay, overwrite=True)

#%%
#mag = 6
predslos = fits.getdata('/users/xliu/Dropbox/expout/predslos_0.0067_v=15_mag=%d_noisefree_0606.fits'%mag)
#
#delayslos = fits.getdata('/users/xliu/Dropbox/expout/delayslos_0.0067_v=15_mag=%d.fits'%mag)
#trueslos = fits.getdata('/users/xliu/Dropbox/expout/trueslos_0.0067_v=15_mag=%d.fits'%mag)
#truth = fits.getdata('/users/xliu/Dropbox/expout/truth_0.0067_v=15_mag=%d.fits'%mag)

#remove "noisy" subaps
#predslos = np.delete(predslos, [11,17,18,24,47,53,54,60], 1)
#delayslos = np.delete(delayslos, [11,17,18,24,47,53,54,60], 1)
#trueslos = np.delete(trueslos, [11,17,18,24,47,53,54,60], 1)
#truth = np.delete(truth, [11,17,18,24,47,53,54,60], 1)

#%%
sim = soapy.Sim("sh_77_openloop.yaml")
sim.aoinit()
sim.makeIMat(forceNew=False)
#imat = sim.recon.interaction_matrix
#cmat = sim.recon.control_matrix
# control matrix
iMatShapes = sim.dms[0].iMatShapes
# influence function
mask = sim.mask

#fits.writeto('sh_77_openloop/cmat_noisy_6_300Hz.fits', cmat)

#cmat_noisy = fits.getdata('sh_77_openloop/cMat_noisy_11.fits')
cmat_noisefree = fits.getdata('sh_77_openloop/cMat_noisefree_%d.fits'%mag) #control matrix generated when both RON and photon noise is turned off

#%% 
"""errorbar/scatter plot of zernike breakdown of phase deviation from some truth vs. Z. order"""
steady_index = 20
modenum = 50
zernikes = truth.dot(cmat) #zernike coefficients
rmse_zer = np.sqrt((zernikes**2).mean(axis=0))

zc_delay = (delayslos-truth).dot(cmat)/rmse_zer
zc_pred = (predslos-truth).dot(cmat)/rmse_zer
zc_true = (trueslos-truth).dot(cmat)/rmse_zer

all_delay = []
mean_delay = [] 
stdmean_delay = []
all_pred = []
mean_pred = [] 
stdmean_pred = []
all_true = []
mean_true = [] 
stdmean_true = []
for mode in range(modenum):
    delay = zc_delay[:,mode].copy().reshape((ntest,-1))[:,steady_index:].copy()
    tmp = [np.sqrt(mse(delay[i],np.zeros((seqlen-1-steady_index,)))) for i in range(ntest)]
    all_delay.append(tmp)
    mean_delay.append(lmean(tmp))
    stdmean_delay.append(lstd(tmp))
    pred = zc_pred[:,mode].copy().reshape((ntest,-1))[:,steady_index:].copy()
    tmp = [np.sqrt(mse(pred[i],np.zeros((seqlen-1-steady_index,)))) for i in range(ntest)]
    all_pred.append(tmp)
    mean_pred.append(lmean(tmp))
    stdmean_pred.append(lstd(tmp))
    true = zc_true[:,mode].copy().reshape((ntest,-1))[:,steady_index:].copy()
    tmp = [np.sqrt(mse(true[i],np.zeros((seqlen-1-steady_index,)))) for i in range(ntest)]
    all_true.append(tmp)
    mean_true.append(lmean(tmp))
    stdmean_true.append(lstd(tmp))
    
plt.figure()
plt.errorbar(range(1,modenum+1), mean_delay, stdmean_delay)
plt.errorbar(range(1,modenum+1), mean_pred, stdmean_pred)
plt.errorbar(range(1,modenum+1), mean_true, stdmean_true)
plt.xticks([1,10,20,30])
plt.legend(['delay','prediction','noisy true'], loc='upper left')
plt.xlabel('Zernike order')
plt.ylabel('normalised RMSE over 79 timesteps in steady state')
plt.title('Normalised Zernike breakdown of slope error @800Hz')
plt.savefig('/users/xliu/dropbox/Dell/prediction/error analysis/errorbar of RMSE of Zernikes vs. order(delay, prediction and noisy true slope error)_@800Hz(RON_only).png', dpi=600, overwrite=True)

plt.plot(stdmean_delay)
plt.plot(stdmean_pred)

all_delay = np.array(all_delay).T
all_pred = np.array(all_pred).T
all_true = np.array(all_true).T

plt.scatter(np.repeat(np.arange(1,modenum+1), ntest).reshape((-1,ntest)).T, all_delay, s=0.1)
plt.scatter(np.repeat(np.arange(1,modenum+1), ntest).reshape((-1,ntest)).T, all_pred, s=0.1)
plt.scatter(np.repeat(np.arange(1,modenum+1), ntest).reshape((-1,ntest)).T, all_true, s=0.1)
plt.plot(range(1,modenum+1), mean_delay, color = 'C0')
plt.plot(range(1,modenum+1), mean_pred, color = 'C1')
plt.plot(range(1,modenum+1), mean_true, color = 'C2')
plt.xticks([1,10,20,30,40,50])
plt.legend(['delay','prediction','noisy true'], loc='upper left')
plt.xlabel('Zernike order')
plt.ylabel('normalised RMSE over 79 timesteps in steady state')
plt.title('Scatter plot of normalised Zernike breakdown of slope error \n@83Hz (nmode=50)')
plt.savefig('/users/xliu/dropbox/Dell/prediction/error analysis/scatter of RMSE of Zernikes vs. order(delay, prediction and noisy true slope error)_@83Hz(RON_only)_nmode=50.png', dpi=600, overwrite=True)

#%%
"""errorbar/scatter plot of phase deviation from some truth vs. frame number"""
#zc_delay = (delayslos-truth).dot(cmat)
#zc_pred = (predslos-truth).dot(cmat)
#zc_true = (trueslos-truth).dot(cmat)

zc_delay = delayslos.dot(cmat_noisy) - truth.dot(cmat_noisefree)
zc_pred = predslos.dot(cmat_noisy) - truth.dot(cmat_noisefree)
zc_true = trueslos.dot(cmat_noisy) - truth.dot(cmat_noisefree)

wfe_delay = [dmCmdDiff_to_WFE(zc_delay[i],iMatShapes,mask) for i in range(ntest*(seqlen-1))]
wfe_delay = np.array(wfe_delay).reshape((-1,seqlen-1))
wfe_pred = [dmCmdDiff_to_WFE(zc_pred[i],iMatShapes,mask) for i in range(ntest*(seqlen-1))]
wfe_pred = np.array(wfe_pred).reshape((-1,seqlen-1))
wfe_true = [dmCmdDiff_to_WFE(zc_true[i],iMatShapes,mask) for i in range(ntest*(seqlen-1))]
wfe_true = np.array(wfe_true).reshape((-1,seqlen-1))

plt.errorbar(range(99), wfe_delay.mean(axis=0), wfe_delay.std(axis=0))
plt.errorbar(range(99), wfe_pred.mean(axis=0), wfe_pred.std(axis=0))

plt.scatter(np.repeat(np.arange(2,seqlen+1), ntest).reshape((-1,ntest)), wfe_delay, s=0.1)
plt.scatter(np.repeat(np.arange(2,seqlen+1), ntest).reshape((-1,ntest)).T, wfe_pred, s=0.1)
plt.scatter(np.repeat(np.arange(2,seqlen+1), ntest).reshape((-1,ntest)).T, wfe_true, s=0.1)
plt.plot(range(2,seqlen+1), wfe_delay.mean(0), 'k-', linewidth=1)
plt.plot(range(2,seqlen+1), wfe_pred.mean(0), 'k-.', linewidth=1)
plt.plot(range(2,seqlen+1), wfe_true.mean(0), 'k:', linewidth=1)
plt.xticks([2,10,20,30,40,50,60,70,80,90,100])
plt.yticks([54, 79, 106])
plt.legend(['delay','prediction','noisy true'], loc='upper right')
plt.xlabel('Frame number')
plt.ylabel('RMS phase error (nm)')
plt.title('Reconstructed phase error (nm) \ncompared with noise-free measurements (windV=15m/s)')
plt.savefig('/users/xliu/dropbox/Dell/prediction/error analysis/RMSE of phase compared with noise-free measurements_v=15_mag=6.png', dpi=300, overwrite=True)
            
dic = pickle.load(open('in_case.p', 'rb'))
dic['v=15_with_noise_free_6'] = [wfe_delay, wfe_pred, wfe_true]
pickle.dump(dic, open('in_case.p','wb'))

#%%
"""compare with true phase screens"""

# wind = pickle.load(open('/users/xliu/dropbox/dell/prediction/datagen/wind_velocity_20190219.p','rb'))
# vs = wind['20190224'][0]
# dirs = wind['20190224'][1]

trueslos = trueslos.reshape((ntest,-1))
delayslos = delayslos.reshape((ntest,-1))
predslos = predslos.reshape((ntest,-1))
#truth = truth.reshape((ntest,-1))

wfe_delay = []
wfe_pred = []
wfe_best = []
#wfe_truth = []

#%%
"""
version one: for small whole phase screens
each phase screen is used only once, no need to remember its initial position as it's always [0,0]
"""
#n_test = 1000
#niter = 100
#nmeasurement = 72
#scrnsize = 66
#for k in range(n_test):
#    sim.config.atmos.scrnNames = ['/users/xliu/dropbox/dell/prediction/split_scrn_256/scrn_256_%d.fits'%k]
##    sim.config.atmos.windSpeeds = [vs[k]]
##    sim.config.atmos.windDirs = [dirs[k]]
#    sim.aoinit()
#    sim.makeIMat()
#    print(k)
#    _ = sim.atmos.moveScrns()
#    
#    for j in range(1,niter): 
#        scrn = sim.atmos.moveScrns()
#        
#        wfe_d, _ = residualWFE(1,scrnsize,cmat_noisy,delayslos[k,(j-1)*nmeasurement: j*nmeasurement],iMatShapes,scrn,mask)
#        wfe_delay.extend([wfe_d])
#        
#        wfe_p, _ = residualWFE(1,scrnsize,cmat_noisy,predslos[k,(j-1)*nmeasurement: j*nmeasurement],iMatShapes,scrn,mask)
#        wfe_pred.extend([wfe_p])
#    
#        wfe_b, _ = residualWFE(1,scrnsize,cmat_noisy,trueslos[k,(j-1)*nmeasurement: j*nmeasurement],iMatShapes,scrn,mask)
#        wfe_best.extend([wfe_b])
#        
#        wfe_t, _ = residualWFE(1,scrnsize,cmat_noisefree,truth[k,(j-1)*nmeasurement: j*nmeasurement],iMatShapes,scrn,mask)
#        wfe_truth.extend([wfe_t])

#%%
"""
version two: for large whole phase screens
each phase screen is used multiple times, and the initial loop position is stored beforehand
"""
n_test = 10
n_samp = 100
niter = 100
scrnsize = 66
nmeasurement = 72
xposes = fits.getdata('../../Dropbox/Dell/prediction/datagen/xposes_v=15.fits')
yposes = fits.getdata('../../Dropbox/Dell/prediction/datagen/yposes_v=15.fits')
for k in range(n_test):
    sim.config.atmos.scrnNames = ['../../Dropbox/Dell/prediction/scrns_4096/scrn_%d.fits'%(k+201)]
    sim.aoinit()
    for j in range(n_samp):
#         sim.config.atmos.windSpeeds = [vs[j]]
#         sim.config.atmos.windDirs = [dirs[j]]
#         dirs[j] *= pi/180.
#         windV = (vs[j]*np.array([cos(dirs[j]),sin(dirs[j])])).T
#         windV = windV*looptime/pxscale
#         sim.atmos.windV[0] = windV
#         if windV[0] > 0: 
#             xpos = uniform(0, wholescrnsize-scrnsize-sim.atmos.windV[0][0]*niter)
#         else: 
#             xpos = uniform(-sim.atmos.windV[0][0]*niter, wholescrnsize-scrnsize)
#         if windV[1] > 0: 
#             ypos = uniform(0, wholescrnsize-scrnsize-sim.atmos.windV[0][1]*niter)
#         else: 
#             ypos = uniform(-sim.atmos.windV[0][1]*niter, wholescrnsize-scrnsize)
        ind = k*n_samp + j
        print(ind)
        xpos = xposes[ind]
        ypos = yposes[ind]
        sim.atmos.scrnPos[0] = np.array([xpos,ypos])
        sim.atmos.xCoords[0] = np.arange(scrnsize).astype('float') + xpos
        sim.atmos.yCoords[0] = np.arange(scrnsize).astype('float') + ypos
        sim.makeIMat(forceNew=False)
        _ = sim.atmos.moveScrns()
    
        for i in range(1,niter): 
            scrn = sim.atmos.moveScrns()
            
            wfe_d, _ = residualWFE(1,scrnsize,cmat_noisefree,delayslos[ind,(i-1)*nmeasurement: i*nmeasurement],iMatShapes,scrn,mask)
            wfe_delay.extend([wfe_d])
#            
            wfe_p, _ = residualWFE(1,scrnsize,cmat_noisefree,predslos[ind,(i-1)*nmeasurement: i*nmeasurement],iMatShapes,scrn,mask)
            wfe_pred.extend([wfe_p])
#        
            wfe_b, _ = residualWFE(1,scrnsize,cmat_noisefree,trueslos[ind,(i-1)*nmeasurement: i*nmeasurement],iMatShapes,scrn,mask)
            wfe_best.extend([wfe_b])
            
#            wfe_t, _ = residualWFE(1,scrnsize,cmat_noisefree,truth[ind,(i-1)*nmeasurement: i*nmeasurement],iMatShapes,scrn,mask)
#            wfe_truth.extend([wfe_t])
                
        sim.iters = 0
        sim.slopes[:] = 0

#%%
"""plots for both versions"""        
wfe_delay = np.array(wfe_delay).reshape((ntest, -1))#[:100]
wfe_pred = np.array(wfe_pred).reshape((ntest, -1))#[:100]
wfe_best = np.array(wfe_best).reshape((ntest, -1))#[:100]
#wfe_truth = np.array(wfe_truth).reshape((ntest, -1))
print(wfe_delay.mean(0).mean())
print(wfe_pred.mean(0)[20:].mean())
print(wfe_best.mean(0).mean())
#print(wfe_truth.mean(0).mean())

#%%
N = wfe_delay.shape[0]
# standard errors
std_delay = wfe_delay.std(0, ddof=1)/np.sqrt(N)
std_pred = wfe_pred.std(0, ddof=1)/np.sqrt(N)
std_best = wfe_best.std(0, ddof=1)/np.sqrt(N)

#%%
plt.semilogy(range(2,niter+1), wfe_delay.mean(0), 'C0', linewidth=1)#, wfe_delay.std(0))
plt.fill_between(range(2,niter+1), wfe_delay.mean(0)-std_delay, wfe_delay.mean(0)+std_delay, alpha=0.5)
plt.semilogy(range(2,niter+1), wfe_pred.mean(0), 'C1', linewidth=1)#, wfe_pred.std(0))
plt.fill_between(range(2,niter+1), wfe_pred.mean(0)-std_pred, wfe_pred.mean(0)+std_pred, alpha=0.5)
plt.semilogy(range(2,niter+1), wfe_best.mean(0), 'C2', linewidth=1)#, wfe_best.std(0))
plt.fill_between(range(2,niter+1), wfe_best.mean(0)-std_best, wfe_best.mean(0)+std_best, alpha=0.5)
#plt.plot(range(2,niter+1), wfe_truth.mean(0), 'k--', linewidth=1)#, wfe_truth.std(0))
#plt.fill_between(range(2,niter+1), wfe_truth.mean(0)-wfe_truth.std(0), wfe_truth.mean(0)+wfe_truth.std(0), alpha=0.5)
#
plt.ylim([155, 180])
plt.legend(['Delay','Prediction','True'], loc='center right')#,'noise-free true'])
plt.xlabel('Frame number')
plt.ylabel('RMS phase error (nm)')
plt.xlim([2,100])
plt.xticks([2]+[i for i in range(10,101,10)])
#plt.ylim([235, 260])
major_ticks = range(155,181,5)
plt.gca().set_yticks(major_ticks, minor=False)
plt.minorticks_off()
plt.yticks(major_ticks,['155','160','165','170','175','180'])

##plt.title('Reconstructed phase error (nm) compared with input phase')
plt.title('Mean RMS phase error in an AO loop\n(GSmag=6)')
###plt.tight_layout()
plt.savefig('/users/xliu/dropbox/Dell/prediction/error analysis/RMSE of phase compared with input phase_v=15_mag=6_0325_detailed.pdf')
##
#%%
#dic = {}
#dic = pickle.load(open('wfe_150Hz_0606.p', 'rb'))
dic['v=15_input_phase_%d'%mag] = [wfe_delay, wfe_pred, wfe_best]
#_, _, _, wfe = dic['v=15_input_phase_%d'%mag]

#%%
#dic['v=15_input_phase_%d'%mag].append(wfe_pred)
pickle.dump(dic, open('wfe_150Hz_0606.p', 'wb'))