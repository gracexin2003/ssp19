# aperture photometry code
# Grace Xin
# Due 7/10/19
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, log10

def IncludeAP(intensity, r, centerX, centerY):
    Nap = 0
    for x in range(len(intensity)):
        for y in range(len(intensity)):
            if (sqrt((x-0.5-centerX)**2+(y-0.5-centerY)**2) > r) or (sqrt((x+0.5-centerX)**2+(y+0.5-centerY)**2) > r) or (sqrt((x+0.5-centerX)**2+(y-0.5-centerY)**2) > r) or (sqrt((x-0.5-centerX)**2+(y+0.5-centerY)**2) > r):
                intensity[x, y] = 0
            else:
                Nap += 1
    return Nap, intensity

def ExcludeAP(intensity, r, centerX, centerY):
    Nap = 0
    for x in range(len(intensity)):
        for y in range(len(intensity)):
            if (sqrt((x-0.5-centerX)**2+(y-0.5-centerY)**2) > r) and (sqrt((x+0.5-centerX)**2+(y+0.5-centerY)**2) > r) and (sqrt((x+0.5-centerX)**2+(y-0.5-centerY)**2) > r) and (sqrt((x-0.5-centerX)**2+(y+0.5-centerY)**2) > r):
                intensity[x, y] = 0
            else:
                Nap += 1
    return Nap, intensity

'''
def ExcludeAn(intensity, inR, outR, centerX, centerY):
    Nan = 0
    for x in range(len(intensity)):
        for y in range(len(intensity)):
            if (sqrt((x-0.5-centerX)**2+(y-0.5-centerY)**2) < inR) and (sqrt((x+0.5-centerX)**2+(y+0.5-centerY)**2) < inR) and (sqrt((x+0.5-centerX)**2+(y-0.5-centerY)**2) < inR) and (sqrt((x-0.5-centerX)**2+(y+0.5-centerY)**2) < inR):
                intensity[x, y] = 0
            elif sqrt((x-centerX)**2+(y-centerY)**2) > outR:
                intensity[x, y] = 0
            else:
                Nan += 1
    return Nan, intensity

def IncludeAn(intensity, inR, outR, centerX, centerY):
    Nan = 0
    for x in range(len(intensity)):
        for y in range(len(intensity)):
            if (sqrt((x-0.5-centerX)**2+(y-0.5-centerY)**2) < inR) or (sqrt((x+0.5-centerX)**2+(y+0.5-centerY)**2) < inR) or (sqrt((x+0.5-centerX)**2+(y-0.5-centerY)**2) < inR) or (sqrt((x-0.5-centerX)**2+(y+0.5-centerY)**2) < inR):
                intensity[x, y] = 0
            elif sqrt((x-centerX)**2+(y-centerY)**2) > outR:
                intensity[x, y] = 0
            else:
                Nan += 1
    return Nan, intensity
'''

def Annulus(intensity, inR, outR, centerX, centerY):
    Nan = 0
    for x in range(len(intensity)):
        for y in range(len(intensity)):
            if sqrt((x-centerX)**2+(y-centerY)**2) < inR:
                intensity[x, y] = 0
            elif sqrt((x-centerX)**2+(y-centerY)**2) > outR:
                intensity[x, y] = 0
            else:
                Nan += 1
    return Nan, intensity

def photometry(file, x, y, r, inR, outR, mode):
    im = fits.getdata(file)
    intensity = im[y-outR:y+outR+1, x-outR:x+outR+1]
    centerX = outR
    centerY = outR
    aperture = np.array([])
    Nap = 0
    if mode == 0:
        Nap, aperture = ExcludeAP(intensity.copy(), r, centerX, centerY)
    elif mode == 1:
        Nap, aperture = IncludeAP(intensity.copy(), r, centerX, centerY)

    Nan, annulus = Annulus(intensity.copy(), inR, outR, centerX, centerY)
    
    plt.imshow(aperture)
    plt.gray()
    plt.show()

    plt.imshow(annulus)
    plt.gray()
    plt.show()
    
    skyAvg = np.sum(annulus)/Nan
    print("skyavg")
    print(skyAvg)
    print()
    
    P = np.sum(aperture)
    S = P - Nap*skyAvg
    print(S)
    g = 0.8 #gain
    dark = 10
    read = 11
    RoS = read**2+(g/sqrt(12))**2
    numerator = sqrt(S*g)
    denominator = sqrt(1+(Nap*(1+Nap/Nan)*((skyAvg*g+dark+RoS)/(S*g))))
    SNR = numerator/denominator
    instmag = (-2.5)*log10(S)
    uncertainty = 1.0857/SNR
    print(Nan)
    print(Nap)
    print(SNR)
    print(instmag)
    print(uncertainty)
    ##print(intensity)

filename = "aptest.FIT"
# 489, 292, 5, 8, 13
# mode: 0 = reject borders; 1 = accept borders; 2 = fractional borders
x_coord, y_coord, radius, skyInR, skyOutR = int(input("x: ")), int(input("y: ")), int(input("r: ")), int(input("inR: ")), int(input("outR: "))
mode = 0
photometry(filename, x_coord-1, y_coord-1, radius, skyInR, skyOutR, mode)

'''
pixel_graph = makeCoords(filename, x_coord, y_coord, radius, skyInR, skyOutR)
x, y, sd_x, sd_y = getData(pixel_graph)
# I'm not completely sure why this works: referenced from Kevin
print("(" + str(x+x_coord-radius) + " " + str(890-(y+y_coord-radius)) + ")")
print("std of x = " + str(sd_x))
print("std of y = " + str(sd_y))
'''
