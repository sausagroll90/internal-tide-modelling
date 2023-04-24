import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt
import numpy as np

N = 0.003
H = 4000
omega = 0.00014
f = 0.00005
U_0 = 10000
rhobar = 1025
g = 9.8

hmax = 200
L = 80000
m=4

num_modes=40

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
    A[n-1] = (1j*(m**2)*(pi**2)*T[n-1]*hmax)/((k[n-1]**3)*(L**2) - 4*(m**2)*(pi**2)*k[n-1])
    B[n-1] = -((1j*(m**2)*(pi**2)*T[n-1]*hmax)/((k[n-1]**3)*(L**2) - 4*(m**2)*(pi**2)*k[n-1]))*np.exp(1j*k[n-1]*L)
    C[n-1] = A[n-1] + B[n-1]
    D[n-1] = A[n-1] + B[n-1]*np.exp(-2j*k[n-1]*L)

def psi_n(n, z):
    return ((-1)**n)*cos((n*pi*z)/H)

def phi_n(n, z):
    return ((-1)**n)*(H/(n*pi))*sin(n*pi*z/H)

def phat_n(n, x):
    if x <= 0:
        return C[n-1]*np.exp(-1j*k[n-1]*x)
    elif x > 0 and x < L:
        return A[n-1]*np.exp(1j*k[n-1]*x) + B[n-1]*np.exp(-1j*k[n-1]*x) + ((m*pi*T[n-1]*hmax)/(L*(k[n-1]**2 - 4*(m**2)*pi**2/L**2)))*sin(2*m*pi*x/L)
    elif x >= L:
        return D[n-1]*np.exp(1j*k[n-1]*x)

def pprime_n(n, x, t):
    return np.real(phat_n(n, x)*np.exp(-1j*omega*t))

def pprime(x, z, t):
    total = 0
    for n in range(1, num_modes+1):
        total += pprime_n(n, x, t) * psi_n(n, z)
    return total

def rhoprime_n(n, x, t):
    return ((n**2)*(pi**2)/(g*(N**2)*(H**2)))*pprime_n(n, x, t)

def rhoprime(x, z, t):
    total = 0
    for n in range(1, num_modes+1):
        total+= rhoprime_n(n, x, t) * phi_n(n, z)
    result = total * (N**2)
    return result

def plotpcontour(xsamples, zsamples, width):
    xs = np.linspace(-width, L + width, xsamples)
    zs = np.linspace(-4000, 0, zsamples)
    
    pprimes = np.zeros((zsamples, xsamples))
    pzeros = np.zeros((zsamples, xsamples))
    ps = np.zeros((zsamples, xsamples))
    
    for zm in range(zsamples):
        for xm in range(xsamples):
            pprimes[zm, xm] = pprime(xs[xm], zs[zm], 0)
            pzeros[zm, xm] = -rhobar*g*zs[zm]
    
    ps = pprimes + pzeros
    fig, ax = plt.subplots()
    ax.contour(xs, zs, ps, 10, colors="k")
    kilometres = lambda x, y: str(x/1000)
    ax.xaxis.set_major_formatter(kilometres)
    ax.yaxis.set_major_formatter(kilometres)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")

def plotrhocontour(xsamples, zsamples, width):
    xs = np.linspace(-width, L + width, xsamples)
    zs = np.linspace(-4000, 100, zsamples)
    
    rhoprimes = np.zeros((zsamples, xsamples))
    rhozeros = np.zeros((zsamples, xsamples))
    rhos = np.zeros((zsamples, xsamples))
    
    for zm in range(zsamples):
        for xm in range(xsamples):
            rhoprimes[zm, xm] = rhoprime(xs[xm], zs[zm], 0)
            rhozeros[zm, xm] = -10*zs[zm] #NEEDS ACTUALLY CORRECTING
    
    rhos = rhoprimes + rhozeros
    fig, ax = plt.subplots()
    ax.contour(xs, zs, rhos, 10, colors="k")
    kilometres = lambda x, y: str(x/1000)
    ax.xaxis.set_major_formatter(kilometres)
    ax.yaxis.set_major_formatter(kilometres)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")


#plotpcontour(250, 100, 200000)
plotrhocontour(250, 100, 200000)