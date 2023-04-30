import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt
import numpy as np

N = 0.003
H = 4000
omega = 0.00014
f = 0.00005
U_0 = 10000 #for exaggeration, real life is =0.04
rhobar = 1025
g = 9.8

hmax = 200
L = 30000

num_modes=20

def compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L):
    c = np.zeros(num_modes)
    k = np.zeros(num_modes)
    T = np.zeros(num_modes, dtype="cdouble")
    A = np.zeros(num_modes, dtype="cdouble")
    B = np.zeros(num_modes, dtype="cdouble")
    D = np.zeros(num_modes, dtype="cdouble")
    E = np.zeros(num_modes, dtype="cdouble")
    
    for n in range(1, num_modes+1):
        c[n-1] = (N*H)/(n*pi)
        k[n-1] = sqrt(omega**2 - f**2)/(c[n-1])
        T[n-1] = 2j*rhobar*U_0*(k[n-1]**2)*(c[n-1]**2)/(omega*H)
        A[n-1] = -(1j*T[n-1]/(k[n-1]**3))*(4*hmax/L**2) - (T[n-1]/(k[n-1]**2))*(2*hmax/L)
        B[n-1] = ((1j*T[n-1]/(k[n-1]**3))*(4*hmax/L**2) + (T[n-1]/(k[n-1]**2))*(2*hmax/L))*np.exp(1j*k[n-1]*L)
        D[n-1] = A[n-1] + B[n-1] + (T[n-1]/(k[n-1]**2))*(4*hmax/L)
        E[n-1] = A[n-1] + B[n-1]*np.exp(-2j*k[n-1]*L) - (T[n-1]/(k[n-1]**2))*(4*hmax/L)*np.exp(-1j*k[n-1]*L)
    
    return c, k, T, A, B, D, E

def psi_n(n, z):
    return ((-1)**n)*cos((n*pi*z)/H)

def phi_n(n, z):
    return ((-1)**n)*(H/(n*pi))*sin(n*pi*z/H)

def phat_n(n, x, c, k, T, A, B, D, E):
    if x <= 0:
        return D[n-1]*np.exp(-1j*k[n-1]*x)
    elif x > 0 and x < L:
        return A[n-1]*np.exp(1j*k[n-1]*x) + B[n-1]*np.exp(-1j*k[n-1]*x) + (T[n-1]/(k[n-1]**2))*(4*hmax/L)*(1-2*x/L)
    elif x >= L:
        return E[n-1]*np.exp(1j*k[n-1]*x)

def pprime_n(n, x, t, c, k, T, A, B, D, E):
    return np.real(phat_n(n, x, c, k, T, A, B, D, E)*np.exp(-1j*omega*t))

def pprime(x, z, t, c, k, T, A, B, D, E):
    total = 0
    for n in range(1, num_modes+1):
        total += pprime_n(n, x, t, c, k, T, A, B, D, E) * psi_n(n, z)
    return total

def rhoprime_n(n, x, t, c, k, T, A, B, D, E):
    return ((n**2)*(pi**2)/(g*(N**2)*(H**2)))*pprime_n(n, x, t, c, k, T, A, B, D, E)

def rhoprime(x, z, t, c, k, T, A, B, D, E):
    total = 0
    for n in range(1, num_modes+1):
        total+= rhoprime_n(n, x, t, c, k, T, A, B, D, E) * phi_n(n, z)
    result = total * (N**2)
    return result

def uhat_n(n, x, c, k, T, A, B, D, E):
    c, k, T, A, B, D, E = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
    if x < 0:
        return -(omega*D[n-1]/(rhobar*k[n-1]*c[n-1]**2))*np.exp(-1j*k[n-1]*x)
    elif x >= 0 and x <= L:
        return (omega/(rhobar*(k[n-1]**2)*(c[n-1]**2)))*(k[n-1]*A[n-1]*np.exp(1j*k[n-1]*x) - k[n-1]*B[n-1]*np.exp(-1j*k[n-1]*x) + (1j*T[n-1]/k[n-1]**2)*(8*hmax/L**2))
    elif x > L:
        return (omega*E[n-1]/(rhobar*k[n-1]*c[n-1]**2))*np.exp(1j*k[n-1]*x)

def u_n(n, x, t, c, k, T, A, B, D, E):
    return np.real(uhat_n(n, x, c, k, T, A, B, D, E)*np.exp(-1j*omega*t))

def u(x, z, t, c, k, T, A, B, D, E):
    total = 0
    for n in range(1, num_modes+1):
        total += u_n(n, x, t, c, k, T, A, B, D, E) * psi_n(n, z)
    return total

def plotp(samples, width):
    c, k, T, A, B, D, E = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
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
                ys[i] = pprime(xs[i], z, t*pi/(2*omega), c, k, T, A, B, D, E)
            plt.plot(xs, ys)

def plotu(samples, width):
    c, k, T, A, B, D, E = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
    xs = np.linspace(-width, L + width, samples)
    ys = np.zeros(samples)
    for h in range(5):
        z =  -h*1000
        plt.figure()
        plt.axis([-width, L + width, -0.25, 0.25])
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("h = " + str(z))
        for t in range(4):
            for i in range(len(xs)):
                ys[i] = u(xs[i], z, t*pi/(2*omega), c, k, T, A, B, D, E)
            plt.plot(xs, ys)
            
def plotpcontour(xsamples, zsamples, width):
    c, k, T, A, B, D, E = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
    xs = np.linspace(-width, L + width, xsamples)
    zs = np.linspace(-4000, 0, zsamples)
    
    pprimes = np.zeros((zsamples, xsamples))
    pzeros = np.zeros((zsamples, xsamples))
    ps = np.zeros((zsamples, xsamples))
    
    for zm in range(zsamples):
        for xm in range(xsamples):
            pprimes[zm, xm] = pprime(xs[xm], zs[zm], 0, c, k, T, A, B, D, E)
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
    c, k, T, A, B, D, E = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, L)
    xs = np.linspace(-width, L + width, xsamples)
    zs = np.linspace(-4000, 0, zsamples)
    
    rhoprimes = np.zeros((zsamples, xsamples))
    rhozeros = np.zeros((zsamples, xsamples))
    rhos = np.zeros((zsamples, xsamples))
    
    for zm in range(zsamples):
        for xm in range(xsamples):
            rhoprimes[zm, xm] = rhoprime(xs[xm], zs[zm], 0, c, k, T, A, B, D, E)
            rhozeros[zm, xm] = -10*zs[zm] #NEEDS ACTUALLY CORRECTING
    
    rhos = rhoprimes + rhozeros
    fig, ax = plt.subplots()
    ax.contour(xs, zs, rhos, 10, colors="k")
    kilometres = lambda x, y: str(x/1000)
    ax.xaxis.set_major_formatter(kilometres)
    ax.yaxis.set_major_formatter(kilometres)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")

def energy_flux(x, t, c, k, T, A, B, D, E):
    total = 0
    for n in range(1, num_modes+1):
        total += u_n(n, x, t, c, k, T, A, B, D, E)*pprime_n(n, x, t, c, k, T, A, B, D, E)
    return total

def compute_energy(x, samples, c, k, T, A, B, D, E):
    ts = np.linspace(0, 2*pi/omega, samples)
    energies = np.zeros(samples)
    for i in range(samples):
        energies[i] = energy_flux(x, ts[i], c, k, T, A, B, D, E)
    energy = (omega*H/(4*pi))*np.trapz(energies, x=ts)
    return energy

def plot_energy_L(Lmin, Lmax, Lsamples, tsamples):
    Ls = np.linspace(Lmin, Lmax, Lsamples)
    ys = np.zeros(Lsamples)
    for i in range(Lsamples):
        c, k, T, A, B, D, E = compute_coefficients(N, H, omega, f, U_0, rhobar, hmax, Ls[i])
        ys[i] = -2*compute_energy(0, tsamples, c, k, T, A, B, D, E)
    plt.plot(Ls, ys)
    
def plot_energy_hmax(hmaxmin, hmaxmax, hmaxsamples, tsamples):
    hmaxs = np.linspace(hmaxmin, hmaxmax, hmaxsamples)
    ys = np.zeros(hmaxsamples)
    for i in range(hmaxsamples):
        c, k, T, A, B, C, D = compute_coefficients(N, H, omega, f, U_0, rhobar, hmaxs[i], L)
        ys[i] = -2*compute_energy(0, tsamples, c, k, T, A, B, C, D)
    plt.plot(hmaxs, ys)


#plotp(500, 300000)
#plotu(500, 300000)
#plotpcontour(250, 100, 200000)
#plotrhocontour(250, 100, 200000)


#plot_energy_L(1000, 1000000, 100, 5)
plot_energy_hmax(0, 4000, 100, 5)
