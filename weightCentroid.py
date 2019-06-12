#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Controiding of ANN weights: find correspondence between ANN neurons and WFS subapertures

"""
import aotools
from aotools.image_processing.centroiders import centreOfGravity

def reimage(x, mask):
    '''
    transfer an input slope vector x onto a 2D WFS plane
    '''
    xind, yind = np.where(mask==1)
    twoDslope_x = np.zeros((mask.shape))
    twoDslope_y = np.zeros((mask.shape))
    twoDslope_x[yind,xind] = x[:len(x)/2]
    twoDslope_y[yind,xind] = x[len(x)/2:]
    return [twoDslope_x, twoDslope_y]

def cog(img):
    '''
    center of gravity of a 2D image
    '''
    import numpy as np
    y_cent,x_cent = np.indices(img.shape) + 1
    y_centroid = (y_cent*img).sum()/img.sum()
    x_centroid = (x_cent*img).sum()/img.sum()
    y_centroid -= 1
    x_centroid -= 1
    return np.array([x_centroid, y_centroid])   

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist   

file_name = '1ly_235_242_0_0_40_128_30_withTT_(lr=0.001)_dp=0-0.1-0-0.1.h5'
model_dp = load_model('/users/xliu/dropbox/expout/'+file_name)
model = load_model('/users/xliu/dropbox/expout/'+file_name)
units = 235
mask = fits.getdata('mask_7.fits')

#model.layers[0].name
weights_dp = model_dp.get_layer('lstm_1').get_weights()[0][:, units*2:units*3]
weights = model.get_layer('lstm_1').get_weights()[0][:, units*2:units*3] # who to who

cent_x = []
cent_y = []
for j in range(235):
    slope_x = reimage(weights_dp[:,j], mask)[0] # x or y
    a, b = cog(abs(slope_x))
    cent_x.append(a)
    cent_y.append(b)

imshow(mask)
plt.scatter(cent_x, cent_y, s=0.5)
#plt.ticks_off()
plt.title('centroid map of FinalHiddenState-OutputXslope weights after training')
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False)         # ticks along the top edge are off
plt.savefig('../error analysis/weight centroid/FinalHiddenState-OutputXslope.png',dpi=600, overwrite=True)

weightDist_dp = []
for j in range(235):
    weightDist_dp.append(calculateDistance(3,3,cent_x[j],cent_y[j]))

plt.hist([weightDist, weightDist_dp], 20)
plt.legend(['before training','after training'])
plt.xlabel('distance of weights centroid from center (m)')
plt.title('distribution of xSlope-CellState weights centroid before and after training')
plt.savefig('../error analysis/weight centroid/histogram_xSlope-CellStateCentroidFromCenter.png',dpi=600, overwrite=True)