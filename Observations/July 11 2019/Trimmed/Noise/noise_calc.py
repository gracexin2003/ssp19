from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

files = ["qf15_obs5_Noise1.fit",
         "qf15_obs5_Noise2.fit",
         "qf15_obs5_Noise3.fit",
         "qf15_obs5_Noise4.fit",
         "qf15_obs5_Noise5.fit",
         "qf15_obs5_Noise6.fit",
         "qf15_obs5_Noise7.fit",
         "qf15_obs5_Noise8.fit",
         "qf15_obs5_Noise9.fit",
         "qf15_obs5_Noise10.fit",
         "qf15_obs5_Noise11.fit",
         "qf15_obs5_Noise12.fit",
         "qf15_obs5_Noise13.fit",
         "qf15_obs5_Noise14.fit",
         "qf15_obs5_Noise15.fit",
         "qf15_obs5_Noise16.fit",
         "qf15_obs5_Noise17.fit",
         "qf15_obs5_Noise18.fit",
         "qf15_obs5_Noise19.fit"]
im1 = fits.getdata(files[0])
im2 = fits.getdata(files[1])
im3 = fits.getdata(files[2])
im4 = fits.getdata(files[3])
im5 = fits.getdata(files[4])
im6 = fits.getdata(files[5])
im7 = fits.getdata(files[6])
im8 = fits.getdata(files[7])
im9 = fits.getdata(files[8])
im10 = fits.getdata(files[9])
im11 = fits.getdata(files[10])
im12 = fits.getdata(files[11])
im13 = fits.getdata(files[12])
im14 = fits.getdata(files[13])
im15 = fits.getdata(files[14])
im16 = fits.getdata(files[15])
im17 = fits.getdata(files[16])
im18 = fits.getdata(files[17])
im19 = fits.getdata(files[18])
print(np.sum(im1+im2+im3+im4+im5+im6+im7+im8+im9+im10+im11+im12+im13+im14+im15+im16+im17+im18+im19)/(19*(len(im1)*len(im1[0]))))
