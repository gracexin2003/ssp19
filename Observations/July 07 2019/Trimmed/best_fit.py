#Create a best-fit line for Cloudy Night 2 data
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
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

#log(dark_current)s
y = [log(np.sum(im1)/(len(im1)*len(im1[0])), e),
     log(np.sum(im2)/(len(im2)*len(im1[0])), e),
     log(np.sum(im3)/(len(im3)*len(im1[0])), e),
     log(np.sum(im4)/(len(im4)*len(im1[0])), e),
     log(np.sum(im5)/(len(im5)*len(im1[0])), e),
     log(np.sum(im6)/(len(im6)*len(im1[0])), e)]
#temp
x = [1/(19.25+273), 1/(13.5+273), 1/(9.1+273), 1/(3.8+273), 1/(-0.9+273), 1/(-5.85+273)]
##x = [19.25, 13.5, 9.1, 3.8, -0.9, -5.85]
#print(x)
#print(y)
print(stats.linregress(x, y))

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
b1err = std_err*(1/(sum([(x[i]-sum(x)/len(x))**2 for i in range(len(x))])))**0.5
b0err = b1err*(sum([x[i]**2 for i in range(len(x))])/len(x))**0.5

print("y = mx + b")
print("m: ", slope)
print("b: ", intercept, " = exponential prefactor")
#print("m err: ", b1err)
#print("b err: ", b0err)
print("r^2: ", r_value**2)
R = 8.3145
k = 1.38*10**(-23)
print("activation energy =", slope*k*-1, slope*R*-1)
print("exponential prefactor =", e**intercept)

plt.plot(x, y, "ro")
x = np.linspace(x[0], x[-1], 100000)
y = slope*x+intercept
plt.plot(x, y, linestyle = "-", label = "Fit line")
plt.title("Cloudy Night #2: Dark Frames vs Time with Constant Temperature")
plt.ylabel("Ln(Pixel Count)")
plt.xlabel("1/Temperature (1/Kelvin)")
plt.show()

y = [np.sum(im1)/(len(im1)*len(im1[0])*120),
     np.sum(im2)/(len(im2)*len(im1[0])*120),
     np.sum(im3)/(len(im3)*len(im1[0])*120),
     np.sum(im4)/(len(im4)*len(im1[0])*120),
     np.sum(im5)/(len(im5)*len(im1[0])*120),
     np.sum(im6)/(len(im6)*len(im1[0])*120)]
#temp
x = [(19.25), (13.5), (9.1), (3.8), (-0.9), (-5.85)]
##x = [19.25, 13.5, 9.1, 3.8, -0.9, -5.85]
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
plt.title("Cloudy Night #2: Dark Current vs Temperature")
plt.ylabel("Pixel Count/sec (120 sec exposure)")
plt.xlabel("Temperature (Celsius)")
plt.show()


y = np.array([np.sum(im1)/(len(im1)*len(im1[0])*120),
              np.sum(im2)/(len(im2)*len(im1[0])*120),
              np.sum(im3)/(len(im3)*len(im1[0])*120),
              np.sum(im4)/(len(im4)*len(im1[0])*120),
              np.sum(im5)/(len(im5)*len(im1[0])*120),
              np.sum(im6)/(len(im6)*len(im1[0])*120)])
#temp
x = np.array([(19.25), (13.5), (9.1), (3.8), (-0.9), (-5.85)])
def exponential_func(x, a, b, c):
    return a*np.exp(-b*x)+c
popt, pcov = curve_fit(exponential_func, x, y, p0=(1, 1e-6, 1))
print(popt, pcov)
xx = np.linspace(x[0], x[-1], 100000)
yy = exponential_func(xx, *popt)
print(exponential_func(xx, *popt))
plt.plot(x,y,'o', xx, yy)
plt.title("Cloudy Night #2: Dark Current vs Temperature (Exponential Fit)")
a, b, c = popt[0], popt[1], popt[2]
plt.legend(('pixel count',
            'fit: a='+str(round(a, 6))+', b='+str(round(b,6))+', c='+str(round(c,6))))
plt.ylabel("Pixel Count/sec (120 sec exposure)")
plt.xlabel("Temperature (Celsius)")
plt.show()
'''
for file in files:
    im = fits.getdata(file)
    for i in range(len(im)):
        for j in range(len(im[0)):
            '''
