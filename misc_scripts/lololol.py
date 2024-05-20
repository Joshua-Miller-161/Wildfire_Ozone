import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())
from misc.misc_utils import Scale, FFT, find_largest_values, MinMaxScale, ButterLowpassFilter

def FFT(t, y, mult_2pi=False):
    # If you believe data is of the form: y = sin(2π * f1 * x) + ...
    #    - set mult_2pi=False to show peak at f1
    # Otherwise if you have: y = sin(f1 * x) + ...
    #    - set mult_2pi=True to show peak at f1
    n = len(t)
    delta = (max(t) - min(t)) / (n-1)
    k = int(n/2)
    f = np.arange(k) / (n*delta)
    Y = abs(np.fft.rfft(y))[:k]
    if not mult_2pi:
        return (f, Y)
    else:
        return (f*(2*np.pi), Y)

def DeNoiseFFT(x, y, num_fft, mult_2pi=False, rescale=True):
    (freq, Y) = FFT(x, Scale(y, y), mult_2pi)
    freq_amps = find_largest_values(Y, freq, num_fft)

    y_new = np.zeros_like(y)
    
    for i in range(num_fft):
        freq = freq_amps[i][0]
        amp  = freq_amps[i][1]
        if mult_2pi:
            y_new += amp * np.sin(freq*x)
        else:
            y_new += amp * np.sin(2*np.pi*freq*x)

    if rescale:
        y_new = MinMaxScale(y_new, min(y), max(y))
    return y_new

fig, ax = plt.subplots(2, 1)
num=100000
t = np.linspace(0, 30, num=num)
y = np.sin(5*t)+np.sin(.1*t)+np.sin(2*t)+np.random.random(num) # If false put in 2*np.pi
(f, Y) = FFT(t, y, True)

new_y = DeNoiseFFT(t, y, 2, True)



ax[0].plot(t, y, 'b-')
ax[1].plot(f, Y)
ax[0].plot(t, new_y, 'r-')



barf = {'a':1, 'b':2}
print(barf)
barf['c'] = 3
print(barf)
plt.show()