import matplotlib.pyplot as plt
import matplotlib.axes as ax
import math
import numpy as np

def f(x):
    return (0.05*math.sin(x) + 3)

def g(x):
    return (-0.2*math.sin(x) + 1.8)

xs = np.linspace(0, 7, 100)
fs = np.zeros(100)
gs = np.zeros(100)

for i in range(100):
    fs[i] = f(xs[i])
    gs[i] = g(xs[i])

plt.plot(xs, fs, color="k")
plt.plot(xs, gs, color="k")
plt.xticks([])
plt.yticks([])
plt.xlabel("(b)")
plt.ylabel("z  ", rotation=0)
plt.xlim(0, 6.3)
plt.ylim(0, 4)