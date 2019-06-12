#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Analyse contribution of different Zernikes modes to uncorrected phase variance in theory and in simulation
Test realiability of simulated Zernike DM 

"""

"""from Noll (1976) Table IV"""
coeff_low = [1.030, 0.582, 0.134, 0.111, 0.088, 0.0648, 0.0587, 0.0525, 0.0463, 0.0401] # first ten modes
coeff_high = [0.2944*i**(-np.sqrt(3)/2) for i in range(11, 52)]
coeff = coeff_low + coeff_high
coeff_diff = np.array([coeff[i] - coeff[i+1] for i in range(50)])

"""simulation"""
D = 4.2
r0 = 0.164
lmda = 500

slopes = fits.getdata('/users/xliu/dropbox/expout/slopes_0.012_RON_512.fits').reshape((-1,72))[:2000].copy()

iMatValue = 500
sim = soapy.Sim("sh_77_openloop.yaml")
sim.aoinit()
sim.makeIMat(forceNew=True)
cmat = sim.recon.control_matrix
zernikes = slopes.dot(cmat) #zernike coefficients
factor = 2*pi/lmda*iMatValue 
#scaling factor for zernike coeffs, so that resulting coeffs can be squared and summed for representing phase variance in rad2
zernikes *= factor
rmse_zer = np.sqrt((zernikes**2).mean(axis=0))
#zer['512_L0=10'] = rmse_zer.copy()
#pickle.dump(zer, open('zernike_breakdown.p','wb'))
#rmse_zer_var = np.sqrt((zernikes**2).std(axis=0))
zer = pickle.load(open('zernike_breakdown.p','rb'))

#rmse_zer = np.sqrt((zernikes**2).sum(axis=1).mean(axis=0))

plt.figure()
plt.plot(np.sqrt(coeff_diff)*(D/r0)**(5/6.), linestyle='dotted', color = 'black') #rmse in theory (rad)
plt.plot(rmse_zer)
#plt.figure()
plt.plot(zer['512'])#,rmse_zer_var) #rmse in simulation (rad)
plt.plot(zer['2048'])
#plt.plot(zer['512_L0=10'])
#plt.plot(zer['512+sh']) # whole scrn size is 512 (telescope=128 pixels) pixels and sub-harmonics added
#plt.plot(zer['2048']) # whole scrn size is 2048 pixels and without sub-harmonics
#plt.plot(zer['2048+sh'])
#plt.plot(zer['4096'])
#plt.plot(zer['4096+sh'])
plt.xticks([0,9,19,29,39,49],['1','10','20','30','40','50'])
plt.xlabel('Zernike order')
plt.ylabel('RMS phase error (rad)')
plt.legend(['theory','simulated-small phase screen (512 pxls)','simulated-larger phase screen (2048 pxls)'])#,'2048','2048+sh','4096','4096+sh'])
plt.title('Zernike breakdown of uncompensated RMS phase error (rad) \nin theory/simulation\n(D=4.2m, r0=0.16m @500nm)')
plt.savefig('../error analysis/zernike_analysis_scrnsize.png',dpi=600)


"""control matrix"""
cmat_512 = fits.getdata('sh_77_openloop/cMat_512.fits')
cmat_2048 = fits.getdata('sh_77_openloop/cMat_2048.fits')
def cov_mx(M):
    return (M.T.dot(M))

imshow(cov_mx(cmat_512))

"""svd of interaction matrix"""
sim = soapy.Sim("sh_77_openloop.yaml")
sim.aoinit()
sim.makeIMat(forceNew=True)
imat = sim.recon.interaction_matrix
cmat = sim.recon.control_matrix

header = fits.Header()
header['TYPE'] = 'Piezo'
header['MODENUM'] = str(64)
#header['IMATNOIS'] = 'True'
#header['NOISETYP'] = 'Poisson'
header['IMATVAL'] = str(500)
#header['SVDCOND'] = str(0.15)
fits.writeto('influence_function_piezo_8x8.fits', iMatShapes, header, overwrite=True)
fits.writeto('control_matrix_piezo_8x8.fits', cmat, header, overwrite=True)

_, s, _ = svd(imat)

plt.plot(range(1,51), s)
plt.xlabel('Mode number')
plt.ylabel('Singular value')
plt.title('SVD of interaction matrix with a 30-mode Zernike DM')
plt.yticks([2, 11])
plt.xticks([1,10,20,30])#,40,50])
plt.savefig('SVD_imat_zernike_30.png', dpi=600, overwrite=True)

imshow(cov_mx(cmat))
covmx = cov_mx(cmat)
plt.plot([covmx[i,i] for i in range(50)])

"""iMatValue test"""
sim = soapy.Sim("sh_77_openloop.yaml")
sim.aoinit()
sim.makeIMat(forceNew=True)
imat = sim.recon.interaction_matrix

response = imat.dot(cmat)
nxactuators = 41
imshow(response, extent=[1,nxactuators,nxactuators,1])
plt.xticks([1,nxactuators])
plt.yticks([1,nxactuators])
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False)         # ticks along the top edge are off
plt.colorbar()
plt.title('calculated DM command in response of Zernike phase input\n(# of Zernikes = {})'.format(nxactuators))
plt.savefig('DM command of Zernike phase input (# of Zernikes = {}).png'.format(nxactuators), dpi=300, overwrite=True)