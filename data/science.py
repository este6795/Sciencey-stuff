import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
k_B = 1.380649e-23  # Boltzmann constant in J/K

# Maxwell-Boltzmann Distribution Fit
def maxwell_boltzmann(E, T):
    """ Maxwell-Boltzmann probability density function """
    return np.sqrt(2 / (np.pi * (k_B * T)**3)) * E**2 * np.exp(-E / (k_B * T))

# Gaussian Fit Function
def gaussian(x, T0, c, sigma):
    """ Gaussian distribution """
    return c * np.exp(-((x - T0)**2) / (2 * sigma**2))

# Load kinetic energies
kinetic_energies = np.loadtxt('myenergy.dat', skiprows=1)[:, -1]

# Load temperature data
temperatures = np.loadtxt('TEMP.dat', skiprows=1)[:, 1]

# Histogram of kinetic energies
hist_kinetic, bin_edges_kinetic = np.histogram(kinetic_energies, bins=30, density=True)
bin_centers_kinetic = 0.5 * (bin_edges_kinetic[1:] + bin_edges_kinetic[:-1])

# Fit Maxwell-Boltzmann to histogram
popt_mb, _ = curve_fit(lambda E, T: maxwell_boltzmann(E, T), bin_centers_kinetic, hist_kinetic, p0=[300])

# Plot Kinetic Energy Histogram
plt.figure(figsize=(10, 6))
plt.hist(kinetic_energies, bins=30, density=True, alpha=0.6, label='Kinetic Energy Data')
plt.plot(bin_centers_kinetic, maxwell_boltzmann(bin_centers_kinetic, popt_mb[0]), 
         color='red', label=f'Maxwell-Boltzmann Fit (T={popt_mb[0]:.2f} K)')
plt.xlabel('Kinetic Energy (kcal/mol)')
plt.ylabel('Probability Density')
plt.title('Maxwell-Boltzmann Distribution Fit')
plt.legend()
plt.grid(True)
plt.savefig('kinetic_energy_histogram.png')
plt.close()

# Histogram of temperatures
hist_temp, bin_edges_temp = np.histogram(temperatures, bins=30, density=True)
bin_centers_temp = 0.5 * (bin_edges_temp[1:] + bin_edges_temp[:-1])

# Fit Gaussian to temperature histogram
popt_gaussian, _ = curve_fit(gaussian, bin_centers_temp, hist_temp, 
                             p0=[np.mean(temperatures), 1, np.std(temperatures)])

# Plot Temperature Histogram
plt.figure(figsize=(10, 6))
plt.hist(temperatures, bins=30, density=True, alpha=0.6, label='Temperature Data')
plt.plot(bin_centers_temp, gaussian(bin_centers_temp, *popt_gaussian), 
         color='red', label=f'Gaussian Fit ($T_0$={popt_gaussian[0]:.2f} K, $\sigma$={popt_gaussian[2]:.2f} K)')
plt.xlabel('Temperature (K)')
plt.ylabel('Probability Density')
plt.title('Gaussian Fit to Temperature Distribution')
plt.legend()
plt.grid(True)
plt.savefig('temperature_histogram.png')
plt.close()
