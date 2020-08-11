 #Create a best-fit line for Cloudy Night 1 data
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from math import log10, sqrt, e

files = []
for i in range(1, 20+1):
    files.append("qf15_obs6_bias"+str(i)+".Combined.fit")

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
im12 = fits.getdata(files[11])
im13 = fits.getdata(files[12])
im14 = fits.getdata(files[13])
im15 = fits.getdata(files[14])
im16 = fits.getdata(files[15])
im17 = fits.getdata(files[16])
im18 = fits.getdata(files[17])
im19 = fits.getdata(files[18])
im20 = fits.getdata(files[19])

#biases
y = [np.sum(im1)/(len(im1)*len(im1[0])),
     np.sum(im2)/(len(im2)*len(im2[0])),
     np.sum(im3)/(len(im3)*len(im3[0])),
     np.sum(im4)/(len(im4)*len(im4[0])),
     np.sum(im5)/(len(im5)*len(im5[0])),
     np.sum(im6)/(len(im6)*len(im6[0])),
     np.sum(im7)/(len(im7)*len(im7[0])),
     np.sum(im8)/(len(im8)*len(im8[0])),
     np.sum(im9)/(len(im9)*len(im9[0])),
     np.sum(im10)/(len(im10)*len(im10[0])),
     np.sum(im11)/(len(im11)*len(im11[0])),
     np.sum(im12)/(len(im12)*len(im12[0])),
     np.sum(im13)/(len(im13)*len(im13[0])),
     np.sum(im14)/(len(im14)*len(im14[0])),
     np.sum(im15)/(len(im15)*len(im15[0])),
     np.sum(im16)/(len(im16)*len(im16[0])),
     np.sum(im17)/(len(im17)*len(im17[0])),
     np.sum(im18)/(len(im18)*len(im18[0])),
     np.sum(im19)/(len(im19)*len(im19[0])),
     np.sum(im20)/(len(im20)*len(im20[0]))]

#temp
x = [23.9, 22.2, 19.5, 18, 15.9, 13.75, 12.1, 10.2, 8, 5.8,
     4, 1.9, 0.2, -2.3, -4, -5.6, -8, -9.8, -12, -14.2]
#power = [0, 6-7, 18, 20, 17, 21, 24, 28, 29, 34-38,
#         39, 44, 48, 54, 55, 6067, 70, 81]
#print(x)
#print(y)
print(stats.linregress(x, y))

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
plt.title("Cloudy Night 3: Biases vs Temperature")
plt.ylabel("Pixel Count")
plt.xlabel("Temperature (Celsius)")
plt.show()


y = [np.sum(im1)/(len(im1)*len(im1[0])),
     np.sum(im2)/(len(im2)*len(im2[0])),
     np.sum(im3)/(len(im3)*len(im3[0])),
     np.sum(im4)/(len(im4)*len(im4[0])),
     np.sum(im5)/(len(im5)*len(im5[0])),
     np.sum(im6)/(len(im6)*len(im6[0])),
     np.sum(im7)/(len(im7)*len(im7[0])),
     np.sum(im8)/(len(im8)*len(im8[0])),
     np.sum(im9)/(len(im9)*len(im9[0])),
     np.sum(im10)/(len(im10)*len(im10[0])),
     np.sum(im11)/(len(im11)*len(im11[0])),
     np.sum(im12)/(len(im12)*len(im12[0])),
     np.sum(im13)/(len(im13)*len(im13[0])),
     np.sum(im14)/(len(im14)*len(im14[0])),
     np.sum(im15)/(len(im15)*len(im15[0])),
     np.sum(im16)/(len(im16)*len(im16[0])),
     np.sum(im17)/(len(im17)*len(im17[0])),
     np.sum(im18)/(len(im18)*len(im18[0])),
     np.sum(im19)/(len(im19)*len(im19[0])),
     np.sum(im20)/(len(im20)*len(im20[0]))]

#temp
x = np.array([23.9, 22.2, 19.5, 18, 15.9, 13.75, 12.1, 10.2, 8, 5.8,
     4, 1.9, 0.2, -2.3, -4, -5.6, -8, -9.8, -12, -14.2])

def exponential_func(x, a, b, c):
    return a*np.exp(-b*x)+c
popt, pcov = curve_fit(exponential_func, x, y, p0=(1, 1e-6, 1))
print(popt, pcov)
xx = np.linspace(x[0], x[-1], 100000)
yy = exponential_func(xx, *popt)
print(exponential_func(xx, *popt))
plt.plot(x,y,'o', xx, yy)
plt.title("Cloudy Night #3:  Biases vs Temperature (Exponential Fit)")
a, b, c = popt[0], popt[1], popt[2]
plt.legend(('pixel count',
            'fit: a='+str(round(a, 6))+', b='+str(round(b,6))+', c='+str(round(c,6))))
plt.ylabel("Pixel Count")
plt.xlabel("Temperature (Celsius)")
plt.show()
