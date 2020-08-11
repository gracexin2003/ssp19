# Calculate noise images for Cloudy Nights
#from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

files = []
for i in range(1, 20+1):
    files.append("qf15_obs6_bias"+str(i)+".Combined.fit")

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
im20 = fits.getdata(files[19])

noise1 = (im1+im2)/2
noise2 = (im2+im3)/2
noise3 = (im3+im4)/2
noise4 = (im4+im5)/2
noise5 = (im5+im6)/2
noise6 = (im6+im7)/2
noise7 = (im7+im8)/2
noise8 = (im8+im9)/2
noise9 = (im9+im10)/2
noise10 = (im10+im11)/2
noise11 = (im11+im12)/2
noise12 = (im12+im13)/2
noise13 = (im13+im14)/2
noise14 = (im14+im15)/2
noise15 = (im15+im16)/2
noise16 = (im16+im17)/2
noise17 = (im17+im18)/2
noise18 = (im18+im19)/2
noise19 = (im19+im20)/2

newIm = fits.PrimaryHDU(noise1)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise1.fit")

newIm = fits.PrimaryHDU(noise2)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise2.fit")

newIm = fits.PrimaryHDU(noise3)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise3.fit")

newIm = fits.PrimaryHDU(noise4)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise4.fit")

newIm = fits.PrimaryHDU(noise5)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise5.fit")

newIm = fits.PrimaryHDU(noise6)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise6.fit")

newIm = fits.PrimaryHDU(noise7)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise7.fit")

newIm = fits.PrimaryHDU(noise8)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise8.fit")

newIm = fits.PrimaryHDU(noise9)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise9.fit")

newIm = fits.PrimaryHDU(noise10)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise10.fit")

newIm = fits.PrimaryHDU(noise11)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise11.fit")

newIm = fits.PrimaryHDU(noise12)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise12.fit")

newIm = fits.PrimaryHDU(noise13)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise13.fit")

newIm = fits.PrimaryHDU(noise14)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise14.fit")

newIm = fits.PrimaryHDU(noise15)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise15.fit")

newIm = fits.PrimaryHDU(noise16)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise16.fit")

newIm = fits.PrimaryHDU(noise17)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise17.fit")

newIm = fits.PrimaryHDU(noise18)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise18.fit")

newIm = fits.PrimaryHDU(noise19)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs5_Noise19.fit")
