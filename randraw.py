#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Generate training data (sequences of slopes of certain length) from a open-loop AO configuration (sh_77_open.yaml)
Each sequence comes with a different wind velocity
In this configuration, DM, SCI are turned off (0) to accelerate data generation as we're only interested in slopes
Number of turbulence layers is set to 1, while multiple layers with customised wind velocity can be set up by removing comments in this code

"""


sim = soapy.Sim("../soapy-master/conf/sh_77_open.yaml")
sim.aoinit()

scrnsize = 512 
nb = 1000
seqlen = 30
nlayer = 1
allslopes = []

vs = uniform(5,10,nb)
dirs = uniform(0,360,nb)
#dirs = dirs.reshape((nb,nlayer))
#strengths = np.zeros((nb,nlayer))
#strengths[:,0] = uniform(0.1,0.9,nb)
#strengths[:,1] = 1-strengths[:,0]

for i in range(nb):
    sim.config.atmos.windSpeeds = [vs[i]]
    sim.config.atmos.windDirs = [dirs[i]]
#    sim.config.atmos.scrnStrengths = strengths[i]
#    windV = (vs[i]*np.array([cos(dirs[i]),sin(dirs[i])])).T
#    windV = windV*looptime/pxscale
#    sim.atmos.windV[0] = windV
    sim.aoinit()
    sim.makeIMat()
    sim.aoloop()
    allslopes.append(sim.allSlopes)


header = fits.Header()
header["r0"] = str(0.16)
header["ITERS"] = str(1000)
header['SEQLEN'] = str(200)
header['ITERTIME'] = str(0.012)
header['NLAYERS'] = str(1)
header['MAG'] = str(10)
header['RON'] = str(1)
header['SVDCOND'] = str(0.15)
fits.writeto("trnslos_20181024/trnslos_20181024_12.fits",np.array(allslopes),header,overwrite=True)

