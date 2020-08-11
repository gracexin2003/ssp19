import numpy as np
import math
from astropy.io import fits
import matplotlib.pyplot as plt

im1 = fits.getdata("qf15_obs1_seq1.00000001.Entered Coordinates.fit")
im2 = fits.getdata("qf15_obs1_seq3.00000007.Entered Coordinates.fit")
plt.gray()
dark_sum = fits.getdata("qf15_obs1_dark.00000001.Entered Coordinates.Dark.fit")
#for i in range(2, 10):
#    dark_sum += fits.getdata("qf15_obs1_dark.0000000" + str(i) + ".Entered Coordinates.Dark.fit")
#dark_sum += fits.getdata("qf15_obs1_dark.00000010.Entered Coordinates.Dark.fit")
#dark_sum += fits.getdata("qf15_obs1_dark.00000011.Entered Coordinates.Dark.fit")
dark_avg = dark_sum[:] / 1
im1 = im1 - dark_avg
im2 = im2 - dark_avg
plt.imshow(im1, vmin = im1.mean(), vmax = 2 * im1.mean())
plt.show()
plt.imshow(im2, vmin = im2.mean(), vmax = 2 * im2.mean())
plt.show()
