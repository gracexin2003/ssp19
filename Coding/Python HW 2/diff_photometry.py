from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, log, log10

def ast_mag(filename, star_x_coord, star_y_coord, star_w, star_mag, ast_x_coord, ast_y_coord, ast_w, blank_x_coord, blank_y_coord):
    im = fits.getdata(filename)
    # extract a box of the star's width centered on its x-y coordinates from the image
    box_r = int(star_w/2)
    star_box = im[star_y_coord-box_r:star_y_coord+box_r+1, star_x_coord-box_r:star_x_coord+box_r+1]
    ##print(star_box)
    # sum the pixel counts of all pixels inside the box ("star+sky")
    star_sky = np.sum(star_box)
    ##print(star_sky)
    # extract a 3X3 box of blank sky centered at the blank x-y coordinates
    box_r = 1
    blank_box = im[blank_y_coord-box_r:blank_y_coord+box_r+1, blank_x_coord-box_r:blank_x_coord+box_r+1]
    ##print(blank_box)
    # determine the average pixel count of all pixels inside that box ("avgSky")
    avgSky = np.sum(blank_box)/9 # total of 9 pixels (3X3)
    # calculate the pixel counts for just the star ("signal")
    n_ap = (star_w)**2
    ##print(n_ap)
    star_sig = star_sky-avgSky*n_ap
    ##print(star_sig)
    # determine constant value from the equation mag = -2.5*log(signal, 10)+const
    const = 2.5*log(star_sig, 10) + star_mag
    ##print(const)
    ##print("")

    # extract a box of the asteroid's width centered on its x-y coordinates
    box_r = int(ast_w/2)
    ast_box = im[ast_y_coord-box_r:ast_y_coord+box_r+1, ast_x_coord-box_r:ast_x_coord+box_r+1]
    ##print(ast_box)
    # sum asteroid+sky
    ast_sky = np.sum(ast_box)
    ##print(ast_sky)
    # 3X3 box of blank sky is the same as before
    # avgSky is the same as before
    # calculate the asteroid's signal
    n_ap = (ast_w)**2
    ##print(n_ap)
    ast_sig = ast_sky-avgSky*n_ap
    ##print(ast_sig)
    
    # find asteroid's magnitude
    mag = -2.5*log(ast_sig, 10)+const
    return mag

# 18.5008
print(ast_mag("sampleimage.fits", 173, 342, 5, 15.26, 351, 154, 3, 200, 200))
# 18.7534
print(ast_mag("sampleimage.fits", 355, 285, 5, 16.11, 351, 154, 3, 200, 200))
