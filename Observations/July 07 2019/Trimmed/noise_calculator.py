# Calculate noise images for Cloudy Nights
#from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

files = ["qf15_obs5_dark1.Combined.fit",
         "qf15_obs5_dark2.Combined.fit",
         "qf15_obs5_dark3.Combined.fit",
         "qf15_obs5_dark4.Combined.fit",
         "qf15_obs5_dark5.Combined.fit",
         "qf15_obs5_dark6.Combined.fit"]

im1 = fits.getdata(files[0])
im2 = fits.getdata(files[1])
im3 = fits.getdata(files[2])
im4 = fits.getdata(files[3])
im5 = fits.getdata(files[4])
im6 = fits.getdata(files[5])

noise1 = (im1+im2)/2
noise2 = (im2+im3)/2
noise3 = (im3+im4)/2
noise4 = (im4+im5)/2
noise5 = (im5+im6)/2

newIm = fits.PrimaryHDU(noise1)
saveIm = fits.HDUList([newIm])
saveIm.writeto("Noise\\qf15_obs5_Noise1.fit")

newIm = fits.PrimaryHDU(noise2)
saveIm = fits.HDUList([newIm])
saveIm.writeto("Noise\\qf15_obs5_Noise2.fit")

newIm = fits.PrimaryHDU(noise3)
saveIm = fits.HDUList([newIm])
saveIm.writeto("Noise\\qf15_obs5_Noise3.fit")

newIm = fits.PrimaryHDU(noise4)
saveIm = fits.HDUList([newIm])
saveIm.writeto("Noise\\qf15_obs5_Noise4.fit")

newIm = fits.PrimaryHDU(noise5)
saveIm = fits.HDUList([newIm])
saveIm.writeto("Noise\\qf15_obs5_Noise5.fit")

'''
plt.imshow(noise1, vmin = noise1.mean(), vmax = 2*noise1.mean())
plt.gray()
plt.show()

plt.imshow(noise2, vmin = noise2.mean(), vmax = 2*noise2.mean())
plt.gray()
plt.show()

plt.imshow(noise3, vmin = noise3.mean(), vmax = 2*noise3.mean())
plt.gray()
plt.show()

plt.imshow(noise4, vmin = noise4.mean(), vmax = 2*noise4.mean())
plt.gray()
plt.show()
'''
