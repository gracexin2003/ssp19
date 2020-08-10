# Combine trimmed Cloudy Night dark images
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

file = input("file: ") #dark
files = ["qf15_obs4_"+file+".00000001.Trimmed.Bias.fit",
         "qf15_obs4_"+file+".00000002.Trimmed.Bias.fit",
         "qf15_obs4_"+file+".00000003.Trimmed.Bias.fit",
         "qf15_obs4_"+file+".00000004.Trimmed.Bias.fit",
         "qf15_obs4_"+file+".00000005.Trimmed.Bias.fit"]

im1 = fits.getdata(files[0])
im2 = fits.getdata(files[1])
im3 = fits.getdata(files[2])
im4 = fits.getdata(files[3])
im5 = fits.getdata(files[4])

finalIm = (im1+im2+im3+im4+im5)/5

newIm = fits.PrimaryHDU(finalIm)
saveIm = fits.HDUList([newIm])
saveIm.writeto("qf15_obs4_"+file+".Combined.Bias.fit")
