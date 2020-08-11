# Combine trimmed Cloudy Night dark images
#from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

for j in range (1, 20+1):
    files = []
    for i in range(1, 11+1):
        intstr = str(i)
        if int(i/10) == 0:
            intstr = "0" + str(i)
        files.append("qf15_obs6_bias"+str(j)+"Bias000000"+intstr+".fit")
    #print(files)
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
    finalIm = (im1+im2+im3+im4+im5+im6+im7+im8+im9+im10+im11)/11

    newIm = fits.PrimaryHDU(finalIm)
    saveIm = fits.HDUList([newIm])
    saveIm.writeto("qf15_obs6_bias"+str(j)+".Combined.fit")
