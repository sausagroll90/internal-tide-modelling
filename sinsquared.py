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
C = np.zeros(num_modes, dtype="cdouble")
D = np.zeros(num_modes, dtype="cdouble")

for n in range(1, num_modes+1):
    c[n-1] = (N*H)/(n*pi)
    k[n-1] = sqrt(omega**2 - f**2)/(c[n-1])
    T[n-1] = 2j*rhobar*U_0*(k[n-1]**2)*(c[n-1]**2)/(omega*H)
    A[n-1] = (2j*(pi**2)*T[n-1]*hmax)/((k[n-1]**3)*(L**2) - 4*(pi**2)*k[n-1])
    B[n-1] = -((2j*(pi**2)*T[n-1]*hmax)/((k[n-1]**3)*(L**2) - 4*(pi**2)*k[n-1]))*np.exp(1j*k[n-1]*L)
    C[n-1] = A[n-1] + B[n-1]
    D[n-1] = A[n-1] + B[n-1]*np.exp(-2j*k[n-1]*L)

def psi_n(n, z):
    return ((-1)**n)*cos((n*pi*z)/H)

def phat_n(n, x):
    if x <= 0:
        return C[n-1]*np.exp(-1j*k[n-1]*x)
    elif x > 0 and x < L:
        return A[n-1]*np.exp(1j*k[n-1]*x) + B[n-1]*np.exp(-1j*k[n-1]*x) + ((2*pi*T[n-1]*hmax)/(L*(k[n-1]**2 - 4*pi**2/L**2)))*sin(2*pi*x/L)
    elif x >= L:
        return D[n-1]*np.exp(1j*k[n-1]*x)

def pprime_n(n, x, t):
    return np.real(phat_n(n, x)*np.exp(-1j*omega*t))

def pprime(x, z, t):
    total = 0
    for n in range(1, num_modes+1):
        total += pprime_n(n, x, t) * psi_n(n, z)
    return total

def uhat_n(n, x):
    if x <= 0:
        return (omega*C[n-1]/(rhobar*k[n-1]*c[n-1]**2))*np.exp(-1j*k[n-1]*x)
    elif x > 0 and x < L:
        return (omega/(rhobar*(k[n-1]**2)*(c[n-1]**2)))*(-k[n-1]*A[n-1]*np.exp(1j*k[n-1]*x) + k[n-1]*B[n-1]*np.exp(-1j*k[n-1]*x) + (4j*(pi**2)*T[n-1]*hmax/((L**2)*(k[n-1]**2) - 4*pi**2))*cos(2*pi*x/L))
    elif x >= L:
        return -(omega*D[n-1]/(rhobar*k[n-1]*c[n-1]**2))*np.exp(1j*k[n-1]*x)

def u_n(n, x, t):
    return np.real(uhat_n(n, x) * np.exp(-1j*omega*t))

def u(x, z, t):
    total = 0
    for n in range(1, num_modes+1):
        total += u_n(n, x, t) * psi_n(n, z)
    return total

samples = 500
width = 300000
xs = np.linspace(-width, L + width, samples)
ys = np.zeros(samples)

for h in range(5):
    z =  -h*1000
    plt.figure()
    plt.axis([-width, L + width, -100, 100])
    plt.xlabel("x")
    plt.ylabel("p\'")
    plt.title("h = " + str(z))
    for t in range(4):
        for i in range(len(xs)):
            ys[i] = pprime(xs[i], z, t*pi/(2*omega))
        plt.plot(xs, ys)

for h in range(5):
    z =  -h*1000
    plt.figure()
    plt.axis([-width, L + width, -0.3, 0.3])
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("h = " + str(z))
    for t in range(4):
        for i in range(len(xs)):
            ys[i] = u(xs[i], z, t*pi/(2*omega))
        plt.plot(xs, ys)