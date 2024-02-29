import numpy as np
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(2, 1)
num=10000000
t = np.linspace(0, 30, num=num)
y = np.sin(5*t) + np.sin(.75*t)+np.random.random(num) # If false put in 2*np.pi
(f, Y) = FFT(t, y, True)

ax[0].plot(t, y)
ax[1].plot(f, Y)

plt.show()