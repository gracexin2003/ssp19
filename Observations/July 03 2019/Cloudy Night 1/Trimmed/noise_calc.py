from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from math import sqrt, log10

files = ["Noise\\qf15_obs4_Noise1.fit",
         "Noise\\qf15_obs4_Noise2.fit",
         "Noise\\qf15_obs4_Noise3.fit"]
im1 = fits.getdata(files[0])
im2 = fits.getdata(files[1])
im3 = fits.getdata(files[2])
print(np.sum(im1)/(len(im1)*len(im1[0])))
print(np.sum(im2)/(len(im1)*len(im1[0])))
print(np.sum(im3)/(len(im1)*len(im1[0])))
print()
print(np.sum(im1+im2+im3)/(3*(len(im1)*len(im1[0]))))
