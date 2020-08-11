from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

files = ["qf15_obs5_Noise1.fit",
         "qf15_obs5_Noise2.fit",
         "qf15_obs5_Noise3.fit",
         "qf15_obs5_Noise4.fit",
         "qf15_obs5_Noise5.fit"]
im1 = fits.getdata(files[0])
im2 = fits.getdata(files[1])
im3 = fits.getdata(files[2])
im4 = fits.getdata(files[3])
im5 = fits.getdata(files[4])
print(np.sum(im1+im2+im3+im4+im5)/(5*(len(im1)*len(im1[0]))))
