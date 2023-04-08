import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt
import numpy as np

N = 0.003
H = 4000
omega = 0.00014
f = 0.00005
U_0 = 0.04
rhobar = 1025

hmax = 500
L = 10000

num_modes=20

c = np.zeros(num_modes)
k = np.zeros(num_modes)
T = np.zeros(num_modes, dtype="cdouble")
A = np.zeros(num_modes, dtype="cdouble")
B = np.zeros(num_modes, dtype="cdouble")

# for n in range(1, num_modes+1):
#     c[n-1] = ((N**2)*(H**2))/((n**2)*(pi**2))
#     k[n-1] = sqrt(omega**2 - f**2)/(c[n-1])
#     T[n-1] = complex(0, (8*rhobar*U_0*hmax*(k[n-1]**2)*(c[n-1]**2))/(omega*H*L))
#     A[n-1] = -1j*T[n-1]/((k[n-1]**3)*L) - T[n-1]/(2*k[n-1]**2)
#     B[n-1] = (1j*T[n-1]/((k[n-1]**3)*L) + T[n-1]/(2*k[n-1]**2))*np.exp(1j*k[n-1]*L)

for n in range(1, num_modes+1):
    c[n-1] = (N*H)/(n*pi)
    k[n-1] = sqrt(omega**2 - f**2)/(c[n-1])
    T[n-1] = 2j*rhobar*U_0*(k[n-1]**2)*(c[n-1]**2)/(omega*H)
    A[n-1] = -(1j*T[n-1]/(k[n-1]**3))*(4*hmax/L**2) - (T[n-1]/(k[n-1]**2))*(2*hmax/L)
    B[n-1] = ((1j*T[n-1]/(k[n-1]**3))*(4*hmax/L**2) + (T[n-1]/(k[n-1]**2))*(2*hmax/L))*np.exp(1j*k[n-1]*L)

def psi_n(n, z):
    return ((-1)**n)*cos((n*pi*z)/H)

def phat_n(n, x):
    if x < 0:
        return (A[n-1] + B[n-1] + (T[n-1]/(k[n-1]**2))*(4*hmax/L))*np.exp(-1j*k[n-1]*x)
    elif x >= 0 and x <= L:
        return A[n-1]*np.exp(1j*k[n-1]*x) + B[n-1]*np.exp(-1j*k[n-1]*x) + (T[n-1]/(k[n-1]**2))*(4*hmax/L)*(1-2*x/L)
    elif x > L:
        return (A[n-1] + B[n-1]*np.exp(-2j*k[n-1]*L) - (T[n-1]/(k[n-1]**2))*(4*hmax/L)*np.exp(-1j*k[n-1]*L))*np.exp(1j*k[n-1]*x)

def pprime_n(n, x, t):
    return np.real(phat_n(n, x)*np.exp(-1j*omega*t))

def pprime(x, z, t):
    total = 0
    for n in range(1, num_modes+1):
        total += pprime_n(n, x, t) * psi_n(n, z)
    return total

samples = 500
xs = np.linspace(-300000, 310000, samples)
ys = np.zeros(samples)

for t in range(4):
    for i in range(len(xs)):
        ys[i] = pprime(xs[i], -4000, t*pi/(2*omega))
    plt.plot(xs, ys)
plt.axis([-300000, 310000, -50, 50])
