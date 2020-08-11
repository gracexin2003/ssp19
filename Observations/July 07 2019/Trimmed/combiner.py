# Combine trimmed Cloudy Night dark images
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

for i in range(1, 6+1):
    file = "dark" + str(i)
    files = ["qf15_obs5_"+file+".00000001.Trimmed.fit",
             "qf15_obs5_"+file+".00000002.Trimmed.fit",
             "qf15_obs5_"+file+".00000003.Trimmed.fit",
             "qf15_obs5_"+file+".00000004.Trimmed.fit",
             "qf15_obs5_"+file+".00000005.Trimmed.fit"]

    im1 = fits.getdata(files[0])
    im2 = fits.getdata(files[1])
    im3 = fits.getdata(files[2])
    im4 = fits.getdata(files[3])
    im5 = fits.getdata(files[4])

    finalIm = (im1+im2+im3+im4+im5)/5

    newIm = fits.PrimaryHDU(finalIm)
    saveIm = fits.HDUList([newIm])
    saveIm.writeto("qf15_obs5_"+file+".Combined.fit")

    file = "bias" + str(i)
    files = ["qf15_obs5_"+file+".00000001.Trimmed.fit",
             "qf15_obs5_"+file+".00000002.Trimmed.fit",
             "qf15_obs5_"+file+".00000003.Trimmed.fit",
             "qf15_obs5_"+file+".00000004.Trimmed.fit",
             "qf15_obs5_"+file+".00000005.Trimmed.fit"]

    im1 = fits.getdata(files[0])
    im2 = fits.getdata(files[1])
    im3 = fits.getdata(files[2])
    im4 = fits.getdata(files[3])
    im5 = fits.getdata(files[4])

    finalIm = (im1+im2+im3+im4+im5)/5

    newIm = fits.PrimaryHDU(finalIm)
    saveIm = fits.HDUList([newIm])
    saveIm.writeto("qf15_obs5_"+file+".Combined.fit")
