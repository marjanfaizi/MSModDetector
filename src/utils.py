#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 2021

@author: Marjan Faizi
"""


import sys
import numpy as np
from scipy import optimize
import pyopenms
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.2)

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
    
    
def multi_gaussian(x, amplitude, mean, stddev):
    y = np.zeros(len(x))
    for i in range(len(mean)):
        y += gaussian(x, amplitude[i], mean[i], stddev[i])

    return y  


def logistic_func(x, max_val, steepness, midpoint):
    return max_val / (1 + np.exp(-steepness*(x-midpoint)))


def fit_logistic_func(x, y, init_val):
    optimized_param, _ = optimize.curve_fit(logistic_func, x, y, maxfev=10000000, p0=init_val)
    return optimized_param

def mapping_mass_to_stddev(mass):
    max_val = 3.6011
    steepness = 2.7585e-05
    midpoint = 39195.1287
    stddev = logistic_func(mass, max_val, steepness, midpoint)
    return stddev

def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 


def get_theoretical_isotope_distribution(sequence, isotope_pattern_resolution):
    molecular_formula = sequence.getFormula()
    isotope_pattern_generator = pyopenms.CoarseIsotopePatternGenerator(isotope_pattern_resolution) 
    isotopes = molecular_formula.getIsotopeDistribution(isotope_pattern_generator)
    isotopes_aslist = []
    for iso in isotopes.getContainer():
        mass = iso.getMZ()
        intensity = iso.getIntensity()
        isotopes_aslist.append([mass, intensity])
        
    isotopes_asarray = np.asarray(isotopes_aslist)
    return isotopes_asarray   


def estimate_stddev_from_isotope_distribution(distribution):
    masses = distribution[:,0]
    intensities = distribution[:,1]

    index_most_abundant_peak = intensities.argmax()
    initial_amplitude = intensities[index_most_abundant_peak]
    intial_stddev = 1.0
    initial_mean = masses[index_most_abundant_peak]
    
    optimized_param, _ = optimize.curve_fit(gaussian, masses, intensities, maxfev=10000000,
                                            p0=[initial_amplitude, initial_mean, intial_stddev])
            
    stddev = optimized_param[2]
    return stddev
  

def determine_unmodified_species_mass(means, species_mass, mass_tolerance):
    masses_in_interval = means[(means >= species_mass-mass_tolerance) & (means <= species_mass+mass_tolerance)]
   
    if len(masses_in_interval) > 1:
        closest_to_species_mass_ix = abs(masses_in_interval - species_mass).argmin()
        unmodified_species_mass = masses_in_interval[closest_to_species_mass_ix]
    elif len(masses_in_interval) == 1:
        unmodified_species_mass = masses_in_interval[0]
    else:
        unmodified_species_mass = 0
    
    return unmodified_species_mass


def plot_spectra(data, peaks, gaussian_model, title, unmodified_species_mass=None):
    x = np.arange(data.search_window_start_mass, data.search_window_end_mass)
    stddev = mapping_mass_to_stddev(gaussian_model.means)
    y = multi_gaussian(x, gaussian_model.amplitudes, gaussian_model.means, stddev)
    plt.figure(figsize=(16,4))
    plt.plot(data.masses, data.intensities, color='gray')
    plt.plot(gaussian_model.means, gaussian_model.amplitudes, '.r')
    plt.plot(peaks[:,0], peaks[:,1], '.b')
    plt.plot(x, y, color='firebrick')
    plt.xlim((data.search_window_start_mass-10, data.search_window_end_mass+10))
    plt.ylim((-0.5, peaks[:,1].max()*1.3))
    plt.title('Sample: '+title+'\n\n')
    plt.xlabel('mass (Da)')
    plt.ylabel('relative intensity (%)')
    plt.legend(['I2MS data', 'Mass shifts', 'Fitted normal distributions'])
    plt.tight_layout()
    sns.despine()
    
    if unmodified_species_mass:
        mass_shifts = gaussian_model.means - unmodified_species_mass
        add_labels(gaussian_model.means, gaussian_model.amplitudes, label=mass_shifts)
    else:
        add_labels(gaussian_model.means, gaussian_model.amplitudes, label=gaussian_model.means)


def add_labels(mean, amplitude, label):   
    for ix in range(len(label)):
        label_str = "{:.2f}".format(label[ix])
        plt.annotate(label_str, (mean[ix], amplitude[ix]), textcoords="offset points", xytext=(5,25), 
                     ha='center', fontsize='small', rotation=45)
        
