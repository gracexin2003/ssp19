from astropy.io import fits
import matplotlib.pyplot as plt
import numpy
from scipy import stats
from math import log, sqrt, e

files = []
for i in range(1, 6+1):
    files.append("qf15_obs5_dark"+str(i)+".Combined.fit")

im1 = fits.getdata(files[0])
im2 = fits.getdata(files[1])
im3 = fits.getdata(files[2])
im4 = fits.getdata(files[3])
im5 = fits.getdata(files[4])
im6 = fits.getdata(files[5])

k = 1.38*10**(-23) #Boltzmann's constant
##lnDe0Matrix = numpy.zeros(shape = (len(im1),len(im1[0]))) 
##deltaEMatrix = numpy.zeros(shape = (len(im1),len(im1[0])))
lnDe0Matrix = []
deltaEMatrix = []
DarkCurrs = []
negSlopes = []
intercepts = []
x = [1/(273+19),
     1/(273+14),
     1/(273+9),
     1/(273+4),
     1/(273-1),
     1/(273-6)]
for r in range(len(im1)):
    if r%10 == 0:
        print(r)
    for c in range(len(im1[0])):
        Deinpix = []
        ##sumDe0 = 0
        ##sumDe0 += im1[r][c]
        Deinpix.append(im1[r][c])
        ##sumDe0 += im2[r][c]
        Deinpix.append(im2[r][c])
        ##sumDe0 += im3[r][c]
        Deinpix.append(im3[r][c])
        ##sumDe0 += im4[r][c]
        Deinpix.append(im4[r][c])
        ##sumDe0 += im5[r][c]
        Deinpix.append(im5[r][c])
        ##sumDe0 += im6[r][c]
        Deinpix.append(im6[r][c])

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, Deinpix)
        negSlopes.append(-slope)
        intercepts.append(intercept)

        deltaE = -1*slope*k
        lnDe0 = intercept

        ##lnDe0Matrix[r][c] = lnDe0
        ##deltaEMatrix[r][c] = deltaE
        lnDe0Matrix.append(lnDe0)
        deltaEMatrix.append(deltaE)

slope, intercept, r_value, p_value, std_err = stats.linregress(deltaEMatrix, lnDe0Matrix)

Emn = 1/slope

Tmn = Emn/k
print(Tmn)

slope, intercept, r_value, p_value, std_err = stats.linregress(negSlopes, intercepts)
plt.plot(negSlopes, intercepts, "ro", markersize=1)
x = numpy.linspace(x[0], x[-1], 100000)
y = slope*x+intercept
plt.plot(x, y, linestyle = "-", label = "Fit line", linewidth = 1)
plt.title("Cloudy Night 2: Pixel Line Intercepts vs Slopes")
plt.ylabel("Intercepts")
plt.xlabel("Slopes")
plt.show()
