import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt
import numpy as np

N = 0.003
H = 4000
omega = 0.00014
f = 0.00005
U_0 = 0.04
rhobar = 1025
g = 9.8

hmax = 500
L = 30000

num_modes=30

def compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L):
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
        A[n-1] = (1j*(pi**2)*T[n-1]*hmax)/((k[n-1]**3)*(L**2) - 4*(pi**2)*k[n-1])
        B[n-1] = -((1j*(pi**2)*T[n-1]*hmax)/((k[n-1]**3)*(L**2) - 4*(pi**2)*k[n-1]))*np.exp(1j*k[n-1]*L)
        C[n-1] = A[n-1] + B[n-1]
        D[n-1] = A[n-1] + B[n-1]*np.exp(-2j*k[n-1]*L)
        
    return c, k, T, A, B, C, D

def psi_n(n, z):
    return ((-1)**n)*cos((n*pi*z)/H)

def phi_n(n, z):
    return ((-1)**n)*(H/(n*pi))*sin(n*pi*z/H)

def phat_n(n, x, c, k, T, A, B, C, D):
    if x <= 0:
        return C[n-1]*np.exp(-1j*k[n-1]*x)
    elif x > 0 and x < L:
        return A[n-1]*np.exp(1j*k[n-1]*x) + B[n-1]*np.exp(-1j*k[n-1]*x) + ((pi*T[n-1]*hmax)/(L*(k[n-1]**2 - 4*pi**2/L**2)))*sin(2*pi*x/L)
    elif x >= L:
        return D[n-1]*np.exp(1j*k[n-1]*x)

def pprime_n(n, x, t, c, k, T, A, B, C, D):
    return np.real(phat_n(n, x, c, k, T, A, B, C, D)*np.exp(-1j*omega*t))

def pprime(x, z, t, c, k, T, A, B, C, D):
    total = 0
    for n in range(1, num_modes+1):
        total += pprime_n(n, x, t, c, k, T, A, B, C, D) * psi_n(n, z)
    return total

def rhoprime_n(n, x, t, c, k, T, A, B, C, D):
    return ((n**2)*(pi**2)/(g*(N**2)*(H**2)))*pprime_n(n, x, t, c, k, T, A, B, C, D)

def rhoprime(x, z, t, c, k, T, A, B, C, D):
    total = 0
    for n in range(1, num_modes+1):
        total+= rhoprime_n(n, x, t, c, k, T, A, B, C, D) * phi_n(n, z)
    result = total * (N**2)
    return result

def uhat_n(n, x, c, k, T, A, B, C, D):
    if x <= 0:
        return -(omega*C[n-1]/(rhobar*k[n-1]*c[n-1]**2))*np.exp(-1j*k[n-1]*x)
    elif x > 0 and x < L:
        return (omega/(rhobar*(k[n-1]**2)*(c[n-1]**2)))*(k[n-1]*A[n-1]*np.exp(1j*k[n-1]*x) - k[n-1]*B[n-1]*np.exp(-1j*k[n-1]*x) - (2j*(pi**2)*T[n-1]*hmax/((L**2)*(k[n-1]**2) - 4*pi**2))*cos(2*pi*x/L))
    elif x >= L:
        return (omega*D[n-1]/(rhobar*k[n-1]*c[n-1]**2))*np.exp(1j*k[n-1]*x)

def u_n(n, x, t, c, k, T, A, B, C, D):
    return np.real(uhat_n(n, x, c, k, T, A, B, C, D) * np.exp(-1j*omega*t))

def u(x, z, t, c, k, T, A, B, C, D):
    total = 0
    for n in range(1, num_modes+1):
        total += u_n(n, x, t, c, k, T, A, B, C, D) * psi_n(n, z)
    return total

def plotp(samples, width):
    c, k, T, A, B, C, D = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
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
                ys[i] = pprime(xs[i], z, t*pi/(2*omega), c, k, T, A, B, C, D)
            plt.plot(xs, ys)

def plotu(samples, width):
    c, k, T, A, B, C, D = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
    xs = np.linspace(-width, L + width, samples)
    ys = np.zeros(samples)
    for h in range(5):
        z =  -h*1000
        plt.figure()
        plt.axis([-width, L + width, -10000, 10000])
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("h = " + str(z))
        for t in range(4):
            for i in range(len(xs)):
                ys[i] = u(xs[i], z, t*pi/(2*omega), c, k, T, A, B, C, D)
            plt.plot(xs, ys)
            
def plotpcontour(xsamples, zsamples, width):
    c, k, T, A, B, C, D = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
    xs = np.linspace(-width, L + width, xsamples)
    zs = np.linspace(-4000, 0, zsamples)
    
    pprimes = np.zeros((zsamples, xsamples))
    pzeros = np.zeros((zsamples, xsamples))
    ps = np.zeros((zsamples, xsamples))
    
    for zm in range(zsamples):
        for xm in range(xsamples):
            pprimes[zm, xm] = pprime(xs[xm], zs[zm], 0, c, k, T, A, B, C, D)
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
    c, k, T, A, B, C, D = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
    xs = np.linspace(-width, L + width, xsamples)
    zs = np.linspace(-4000, 0, zsamples)
    
    rhoprimes = np.zeros((zsamples, xsamples))
    rhozeros = np.zeros((zsamples, xsamples))
    rhos = np.zeros((zsamples, xsamples))
    
    for zm in range(zsamples):
        for xm in range(xsamples):
            rhoprimes[zm, xm] = rhoprime(xs[xm], zs[zm], 0, c, k, T, A, B, C, D)
            rhozeros[zm, xm] = -((rhobar*N**2)/g)*zs[zm]
    
    rhos = rhoprimes + rhozeros
    fig, ax = plt.subplots()
    ax.contour(xs, zs, rhos, 10, colors="k")
    kilometres = lambda x, y: str(x/1000)
    ax.xaxis.set_major_formatter(kilometres)
    ax.yaxis.set_major_formatter(kilometres)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")

def J_n(n, x, c, k, T, A, B, C, D):
    return (H/4)*np.real(phat_n(n, x, c, k, T, A, B, C, D)*np.conjugate(uhat_n(n, x, c, k, T, A, B, C, D)))

def J(x, c, k, T, A, B, C, D):
    total = 0
    for n in range(1, num_modes+1):
        total += J_n(n, x, c, k, T, A, B, C, D)
    return total

def plot_energy_L(Lmin, Lmax, Lsamples):
    Ls = np.linspace(Lmin, Lmax, Lsamples)
    Js = np.zeros(Lsamples)
    for i in range(Lsamples):
        c, k, T, A, B, C, D = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, Ls[i])
        Js[i] = -1*J(-1, c, k, T, A, B, C, D) + J(L+1, c, k, T, A, B, C, D)
    fig, ax = plt.subplots()
    ax.plot(Ls, Js, color="k")
    kilometres = lambda x, y: str(x/1000)
    ax.xaxis.set_major_formatter(kilometres)
    ax.set_xlabel("L (km)")
    ax.set_ylabel("J (W m$^{-1}$)")

def plot_energy_hmax(hmaxmin, hmaxmax, hmaxsamples):
    hmaxs = np.linspace(hmaxmin, hmaxmax, hmaxsamples)
    Js = np.zeros(hmaxsamples)
    for i in range(hmaxsamples):
        c, k, T, A, B, C, D = compute_coefficients(N, H, omega, f, U_0, rhobar, hmaxs[i], L)
        Js[i] = -1*J(-1, c, k, T, A, B, C, D) + J(L+1, c, k, T, A, B, C, D)
    fig, ax = plt.subplots()
    ax.plot(hmaxs, Js, color="k")
    ax.set_xlabel("$h_{max}$ (m)")
    ax.set_ylabel("J (W m$^{-1}$)")
    



#plotp(500, 300000)
#plotu(500, 30000)
#plotpcontour(250, 100, 200000)
#plotrhocontour(250, 100, 200000)

plot_energy_L(1000, 1000000, 100)
plot_energy_hmax(0, 2000, 100)

