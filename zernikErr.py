#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
analyse error in each zernike coefficient as prediction goes on

"""
file_name = '1ly_72_72_20_128_30_withTT(lr=0.001)_gauss_0.1'
model = load_model('/users/xliu/Dropbox/expout/'+file_name+'.h5')
n_test = 1000
ndm = 1
scrnsize = 130
niter = 40

"""generate a Zernike DM"""
sim = soapy.Sim("/users/xliu/desktop/soapy-master/conf/sh_77_openloop.yaml")
sim.aoinit()
sim.makeIMat(forceNew=True)
iMatShapes = sim.dms[0].iMatShapes
cmat = sim.recon.control_matrix

zc_all = [] 
zc_pred = [] 

for k in range(n_test):
    sim = soapy.Sim("/users/xliu/desktop/soapy-master/conf/sh_77_openloop.yaml")
    sim.config.atmos.scrnNames = ['scrns_512_2/scrn_%d.fits'%k]
    sim.aoinit()
    sim.makeIMat(forceNew=False)
    sim.aoloop()
    
    allslopes = sim.allSlopes.copy()
    
    """feed into slopes to generate Zernike DM commands"""
    for j in range(niter-1):       
        zc_all.extend([cmat.T.dot(allslopes[j])])
        inp = allslopes[:j+1,:].reshape((1,j+1,72))
        pred_slos = model.predict(inp).reshape((72,))
        zc_pred.extend([cmat.T.dot(pred_slos)])
    zc_all.extend([cmat.T.dot(allslopes[niter-1])])

zcdic = {}
zcdic['v=5,mag=6,ron=1'] = [zc_all, zc_pred]
pickle.dump(zcdic, open('zernikErr.p', 'wb'))

"""plots of absolute delay and prediction error of a certain mode vs. frame number"""
n_test = 1000
zcdic = pickle.load(open('/users/xliu/dropbox/expout/zernikErr.p','rb'))
zc_all = np.array(zcdic['v=5,mag=6,ron=1'][0])
zc_pred = np.array(zcdic['v=5,mag=6,ron=1'][1])

mode = 49
alll = zc_all[:,mode].copy().reshape((n_test,-1))
pred = zc_pred[:,mode].copy().reshape((n_test,-1))
true = alll[:,1:].copy()
delay = alll[:,:-1].copy()
seqlen = pred.shape[1]
prederr = []
delayerr = []
for i in range(seqlen):
    prederr.append(np.sqrt(mse(true[:,i], pred[:,i])))
    delayerr.append(np.sqrt(mse(true[:,i], delay[:,i])))
plt.plot(prederr)
plt.plot(delayerr)
plt.xticks([0,38],['2','40'])
plt.xlabel('frame number')
plt.ylabel('RMSE')
plt.legend(['prediction','delay'])
#plt.title('Spherical RMSE vs. frame number')

"""
plots of normalised prediction error of the first ten modes vs. frame number
all the lines are supposed to well separate as lower modes usually change slower and might need more time to predict
"""
prederr_all = []
for mode in range(10):
    alll = zc_all[:,mode].copy().reshape((n_test,-1))
    pred = zc_pred[:,mode].copy().reshape((n_test,-1))
    true = alll[:,1:].copy()
    for i in range(seqlen):
        prederr_all.append(np.sqrt(mse(true[:,i], pred[:,i])))
prederrarr = np.array(prederr_all).reshape((10,-1))
errmin = prederrarr.min(axis=1).reshape((-1,1))
errmax = prederrarr.max(axis=1).reshape((-1,1))
normerr = (prederrarr-errmin)/(errmax-errmin)

import matplotlib.pylab as pl
index = [0,2,4,6,8,9]
colors = pl.cm.jet(np.linspace(0,1,len(index)))
for i in range(len(index)):
    plt.plot(normerr[index[i]], color=colors[i])
plt.legend(['Tip','Defocus','Astigmatism','Coma','Trefoil','Spherical'])
plt.xticks([0,8,38],['2','10','40'])
plt.xlabel('frame number')
plt.ylabel('Normalised RMSE')
plt.title('normalised prediction RMSE of lower Zernike modes vs. frame number')
plt.savefig('rmse of different modes vs. frame number.eps')

"""percentage error of each mode in frame 30"""
nb = 28
zc_all_ = zc_all.reshape((1000,-1,50))
zc_pred_ = zc_pred.reshape((1000,-1,50))
true = zc_all_[:,nb+1,:]
pred = zc_pred_[:,nb,:]
moderr = []
modenum = pred.shape[1]
for i in range(modenum):
    moderr.append((abs((true[:,i]-pred[:,i])/true[:,i])).mean())
