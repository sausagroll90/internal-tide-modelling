import matplotlib.pyplot as plt
import matplotlib.axes as ax
import math
import numpy as np

ns = np.arange(40)
A = np.zeros(40)

for n in ns:
    if n > 0:
        A[n] = 1/(ns[n])
    else:
        A[n] = 0

def f(x):
    total = 0.0
    for i in range(len(ns)):
        total += A[i]*math.sin(i*math.pi*x/5)
    return total

print(ns)
print(A)

xs = np.linspace(0, 30, 200)
fs = np.zeros(200)

for n in range(200):
    fs[n] = f(xs[n])

plt.plot(xs, fs)