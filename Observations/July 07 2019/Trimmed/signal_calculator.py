# calculate bias-subtracted dark signals
# from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

avgSig = []
stds = []
prevDark = fits.getdata("qf15_obs5_dark1.Combined.fit")
prevBias = fits.getdata("qf15_obs5_bias1.Combined.fit")
PrevDark = abs(np.sum(prevDark-prevBias))
avgSig.append(PrevDark/(len(prevDark)*len(prevDark[0])))
for i in range(2, 6+1):
    darkfile = "dark" + str(i)
    biasfile = "bias" + str(i)

    dark = fits.getdata("qf15_obs5_"+darkfile+".Combined.fit")
    bias = fits.getdata("qf15_obs5_"+biasfile+".Combined.fit")
    Dark = abs(np.sum(dark-bias))
    
    avgSig.append(Dark/(len(dark)*len(dark[0])))
    stds.append((abs(Dark-PrevDark)/(len(dark)*len(dark[0])))/sqrt(2))
    PrevDark = Dark

signal = sum(avgSig)/6
print("signal: ", signal)
print(stds)

stdCount = ['1', '2', '3', '4', '5']
x = np.arange(len(stdCount))
y = stds.copy()
plt.bar(x, y)
plt.ylabel('Uncertainty')
plt.xticks(x, stdCount)
plt.title('Cloudy Night 2: Uncertainty Trend')

plt.show()
