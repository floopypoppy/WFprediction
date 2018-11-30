#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 5 17:59:53 2018
Necessary libraries used 

"""
from __future__ import division

import pickle
import numpy as np
import soapy
import time
import keras
#import matplotlib
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from numpy import arccos, pi, sin, cos
from pylab import *
from numpy.random import uniform
from numpy.linalg import svd
from keras.models import load_model
from sklearn.metrics import mean_squared_error as mse

rad2arc = 180*3600/pi
arc2rad = 1./rad2arc