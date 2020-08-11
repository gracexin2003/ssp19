# calculate bias-subtracted dark signals
# from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

sigDiff = []
stds = []
prevbias = fits.getdata("qf15_obs6_bias1.Combined.fit")
sigDiff.append(abs(np.sum(prevbias)))
for i in range(2, 20+1):
    biasfile = "bias" + str(i)

    bias = fits.getdata("qf15_obs6_"+biasfile+".Combined.fit")
    sigDiff.append(abs(np.sum(bias)))

    stds.append(abs(np.sum(bias-prevbias)/(len(bias)*len(bias[0])))/sqrt(2))
    prevbias = bias
signal = sum(sigDiff)/((len(bias)*len(bias[0])))/20
print("signal: ", signal)
print(stds)

stdCount = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
     '11', '12', '13', '14', '15', '16', '17', '18', '19']
x = np.arange(len(stdCount))
y = stds.copy()
plt.bar(x, y)
plt.ylabel('Uncertainty')
plt.xticks(x, stdCount)
plt.title('Cloudy Night 3: Uncertainty Trend')

plt.show()
