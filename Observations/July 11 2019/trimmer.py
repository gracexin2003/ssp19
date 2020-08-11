# Trim images for Cloudy Nights
#from PIL import Image
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

#qf15_obs5_cloudynight2_dark4Dark00000005.fit

#file = input("file: ")
for j in range(1, 20+1):
    for i in range(1, 11+1):
        intstr = str(i)
        if int(i/10) == 0:
            intstr = "0" + str(i)
        fileIn = ("qf15_obs6_bias"+str(j)+"Bias000000"+intstr+".fit")
        #print(fileIn)
        
        fileOut = "Trimmed\\qf15_obs6_bias"+str(j)+"Bias000000"+intstr+".Trimmed.fit"
        #print(fileOut)
        im = fits.getdata(fileIn)
        #plt.imshow(im, vmin = im.mean(), vmax = 2*im.mean())
        #plt.gray()
        #plt.show()
        im = trim(fits.getdata(fileIn), fileOut)
        #plt.imshow(im, vmin = im.mean(), vmax = 2*im.mean())
        #plt.gray()
        #plt.show()
        

