from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.transforms as tr
import numpy as np
from math import sqrt

filename = "ds9.seq1.01.fits"
invertX = False
invertY = True

im = fits.getdata(filename)
plt.imshow(im)
plt.gray()
if invertY:
    im = np.flipud(im)
    plt.imshow(im)
if invertX:
    im = np.flip(im)
    plt.imshow(im)
plt.show()

