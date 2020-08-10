# Write the centroid code
# Grace Xin
# Due 7/8/19
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def makeCoords(filename, x_coord, y_coord, radius, inR, outR):
    im = fits.getdata(filename)
    plt.imshow(im, vmin = im.mean(), vmax = 2*im.mean())
    plt.gray()
    plt.show()
    sq_ap_plus_graph = im[y_coord-radius-outR:y_coord+radius+outR+1, x_coord-radius-outR:x_coord+radius+outR+1]
    plt.imshow(sq_ap_plus_graph, vmin = im.mean(), vmax = 2*im.mean())
    plt.gray()
    plt.show()
    return makeCircle(sq_ap_plus_graph, radius, inR, outR)

def makeCircle(intensity, r, inR, outR):
    ##print(intensity)
    centerX = r+outR
    centerY = r+outR
    skyAvg = 0
    skyPixels = 0
    for x in range(len(intensity)):
        for y in range(len(intensity[x])):
            if sqrt((x-r)**2+(y-r)**2) >= inR and sqrt((x-r)**2+(y-r)**2) <= outR:
                skyAvg += intensity[x, y]
                skyPixels += 1
    skyAvg /= skyPixels
    ##print(skyAvg)
    for x in range(len(intensity)):
        for y in range(len(intensity[x])):
            if sqrt((x-r)**2+(y-r)**2) > r:
                intensity[x, y] = 0
    ##print(intensity)
    for x in range(len(intensity)):
        for y in range(len(intensity[x])):
            intensity[x, y] -= skyAvg
            if intensity[x, y] < 0:
                intensity[x, y] = 0
    
    ##print(intensity)
    return intensity
  
def getData(intensity):
    # hand-calculated column sums: [89, 388, 363, 211, 78]
    column_w = []
    for y in range(len(intensity[0])):
        sum = 0
        for x in range(len(intensity)):
            sum += intensity[x, y]
        column_w.append(sum)

    ##print(column_w)
    # hand-calculated row sums:    [95, 187, 383, 338, 127]
    row_w = []
    for x in range(len(intensity)):
        sum = 0
        for y in intensity[x]:
            sum += y
        row_w.append(sum)
    ##print(row_w)
    ##print("")

    ### expected centroid position answers: x = 1.824, y = 2.191

    # hand-calculated column weighted mean: 1.8237378210806
    sum_w, sum = 0, 0
    for x in range(len(column_w)):
        sum_w += column_w[x]*x
        sum += column_w[x]
    col_w_m = sum_w/sum
    # hand-calculated row weighted mean: 2.1902654867257
    sum_w, sum = 0, 0
    for x in range(len(row_w)):
        sum_w += row_w[x]*x
        sum += row_w[x]
    row_w_m = sum_w/sum

    ### expected standard deviation answers: sx = 0.0311, sy = 0.0328

    # hand-calculated standard deviation for x: 0.027109098601881
    diff_w_x = 0
    for x in range(len(column_w)):
        diff_w_x += column_w[x]*(abs(x-col_w_m)**2)
    sx = sqrt(diff_w_x/(sum-1))
    sdev_x = sx/sqrt(sum)
    # hand-calculated standard deviation for y: 0.02682439089053
    diff_w_y = 0
    for x in range(len(row_w)):
        diff_w_y += row_w[x]*(abs(x-row_w_m)**2)
    sy = sqrt(diff_w_y/(sum-1))
    sdev_y = sy/sqrt(sum)
    
    return col_w_m, row_w_m, sdev_x, sdev_y

'''
lick = np.array([[0,  0,   21,  0,  0,  0],
                 [0,  56,  51,  53, 0,  0],
                 [23, 120, 149, 73, 18, 0],
                 [0,  101, 116, 50, 0,  0],
                 [0,  0,   26,  0,  0,  0]])
print(getData(lick))
'''

##im = fits.getdata("sampleimage.fits")
##print(im)

# Centroid = 350.9958, 153.9956
# Uncertainty in x,y: 0.005254018, 0.005249733
x_coord, y_coord, radius, skyInR, skyOutR = 746, 401, 3, 6, 11
# ds9.seq1.01.fits
# qf15_obs4_seq1.00000001.Entered Coordinates.reduced.fits
pixel_graph = makeCoords("qf15_obs4_seq1_01.fits", x_coord, y_coord, radius, skyInR, skyOutR)
x, y, sd_x, sd_y = getData(pixel_graph)
# I'm not completely sure why this works: referenced from Kevin
print("(" + str(x+x_coord-radius) + " " + str(890-(y+y_coord-radius)) + ")")
print("std of x = " + str(sd_x))
print("std of y = " + str(sd_y))
