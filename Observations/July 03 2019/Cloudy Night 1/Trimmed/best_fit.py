#Create a best-fit line for Cloudy Night 1 data
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from math import sqrt, log10

files = ["qf15_obs4_dark.Combined.Dark.fit",
         "qf15_obs4_dark2.Combined.Dark.fit",
         "qf15_obs4_dark3.Combined.Dark.fit",
         "qf15_obs4_dark4.Combined.Dark.fit"]

im1 = fits.getdata(files[0])
im2 = fits.getdata(files[1])
im3 = fits.getdata(files[2])
im4 = fits.getdata(files[3])

#dark_currents
y = [np.sum(im1)/(len(im1)*len(im1[0])),
     np.sum(im2)/(len(im2)*len(im2[0])),
     np.sum(im3)/(len(im3)*len(im3[0])),
     np.sum(im4)/(len(im4)*len(im4[0]))]
#times
#x = [log10(5)**3, log10(10)**3, log10(20)**3, log10(40)**3]
x = [5, 10, 20, 40]
#print(x)
#print(y)
yerr = [abs(y[i]-y[i-1])/sqrt(2) for i in range(1, len(y))]
yerr.append(yerr[-1])
print(yerr)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
b1err = std_err*(1/(sum([(x[i]-sum(x)/len(x))**2 for i in range(len(x))])))**0.5
b0err = b1err*(sum([x[i]**2 for i in range(len(x))])/len(x))**0.5

print("y = mx + b")
print("m: ", slope)
print("b: ", intercept)
#print("m err: ", b1err)
#print("b err: ", b0err)
print("r^2: ", r_value**2)

plt.plot(x, y, "ro")
x = np.linspace(x[0], x[-1], 100000)
y = slope*x+intercept
plt.plot(x, y, linestyle = "-", label = "Fit line")
#plt.errorbar(x, y, yerr=yerr, uplims=True,lolims=True, linestyle="None")
plt.title("Cloudy Night #1: Dark Current vs Time with Constant Temperature")
plt.xlabel("Time (seconds)")
plt.ylabel("Pixel Count")
plt.show()

x = [0, 0, 0, 0]
y = [np.sum(im1)/(len(im1)*len(im1[0])*5),
     np.sum(im2)/(len(im2)*len(im2[0])*10),
     np.sum(im3)/(len(im3)*len(im3[0])*20),
     np.sum(im4)/(len(im4)*len(im4[0])*40)]
plt.plot(x, y, "ro")
plt.title("Cloudy Night #1: Doubling Exposure Time at Constant Temperature")
plt.xlabel("Temperature (Constant at 19 degrees Celsius)")
plt.ylabel("Pixel Count/sec")
plt.show()
