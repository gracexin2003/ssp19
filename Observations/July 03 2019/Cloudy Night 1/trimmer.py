# Trim images for Cloudy Nights
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

def trim(im, filename):
    trimmedIm = im[20:-20, 20:-20]
    newIm = fits.PrimaryHDU(trimmedIm)
    saveIm = fits.HDUList([newIm])
    saveIm.writeto(filename)
    #checkIm = fits.getdata(filename)
    #plt.imshow(checkIm, vmin = im.mean(), vmax = 2*im.mean())
    #plt.show()
    return trimmedIm

file = input("file: ")
fileIn = "qf15_obs4_bias.0000000" + file + ".Entered Coordinates.Bias.fit"
fileOut = "U:\Observations\July 03 2019\Cloudy Night 1\Trimmed\\qf15_obs4_bias.0000000" + file + ".Trimmed.Bias.fit"
im = fits.getdata(fileIn)
#plt.imshow(im, vmin = im.mean(), vmax = 2*im.mean())
#plt.gray()
#plt.show()
im = trim(fits.getdata(fileIn), fileOut)
#plt.imshow(im, vmin = im.mean(), vmax = 2*im.mean())
#plt.gray()
#plt.show()

