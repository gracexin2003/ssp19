# calculate bias-subtracted dark signals
# from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

avgSig = []
stds = []
prevDark = fits.getdata("qf15_obs4_dark.Combined.Dark.fit")
bias = fits.getdata("qf15_obs4_bias.Combined.Bias.fit")
PrevDark = abs(np.sum(prevDark-bias))
avgSig.append(PrevDark/(len(prevDark)*len(prevDark[0])))
for i in range(2, 4+1):
    darkfile = "dark" + str(i)
    biasfile = "bias" + str(i)

    dark = fits.getdata("qf15_obs4_"+darkfile+".Combined.Dark.fit")
    Dark = abs(np.sum(dark-bias))
    
    avgSig.append(Dark/(len(dark)*len(dark[0])))
    stds.append((abs(Dark-PrevDark)/(len(dark)*len(dark[0])))/sqrt(2))
    PrevDark = Dark

signal = sum(avgSig)/4
print("signal: ", avgSig)
print(stds)



stdCount = ['1', '2', '3']
x = np.arange(len(stdCount))
y = stds.copy()
plt.bar(x, y)
plt.ylabel('Uncertainty')
plt.xticks(x, stdCount)
plt.title('Cloudy Night 1: Uncertainty Trend')

plt.show()
