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
    return amplitude * np.exp(-0.5*((x - mean) / stddev)**2)
    

def integral_of_gaussian(amplitude, stddev):
    return np.sqrt(2*np.pi)*amplitude*stddev


def two_gaussian(x, amp1, amp2, mean1, mean2, stddev):
    y = np.zeros(len(x))
    y += gaussian(x, amp1, mean1, stddev) + gaussian(x, amp2, mean2, stddev)
    return y  
 
def multi_gaussian(x, amplitude, mean, stddev):
    y = np.zeros(len(x))
    for i in range(len(mean)):
        y += gaussian(x, amplitude[i], mean[i], stddev)

    return y  


def logistic_func(x, max_val, steepness, midpoint):
    return max_val / (1 + np.exp(-steepness*(x-midpoint)))


def fit_logistic_func(x, y, init_val):
    optimized_param, _ = optimize.curve_fit(logistic_func, x, y, maxfev=10000000, p0=init_val)
    return optimized_param


def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 


def isotope_distribution_fit_par(protein_sequence, isotope_pattern_resolution):
    distribution =  get_theoretical_isotope_distribution(pyopenms.AASequence.fromString(protein_sequence), 
                                                         isotope_pattern_resolution)
    masses = distribution[:,0]
    intensities = distribution[:,1]

    index_most_abundant_peak = intensities.argmax()
    initial_amplitude = intensities[index_most_abundant_peak]
    intial_stddev = 1.0
    initial_mean = masses[index_most_abundant_peak]
    if not np.isnan(masses).all() and not  np.isnan(intensities).all(): 
        optimized_param, _ = optimize.curve_fit(gaussian, masses, intensities, maxfev=10000000,
                                                p0=[initial_amplitude, initial_mean, intial_stddev])
        amplitude = optimized_param[0]
        mean = optimized_param[1]
        stddev = optimized_param[2]
        return mean, stddev, amplitude
    
    else:
        return 0, 0, 0


def read_fasta(fasta_file_name):
    entries = []
    fasta_file = pyopenms.FASTAFile()
    fasta_file.load(fasta_file_name, entries)
    protein_sequences = {}
    for e in entries:
        protein_sequences[e.identifier] = e.sequence
    return protein_sequences


def get_theoretical_isotope_distribution(sequence, isotope_pattern_resolution, modus="coarse"):
    molecular_formula = sequence.getFormula()       
    if modus == "fine":
        isotope_pattern_generator = pyopenms.FineIsotopePatternGenerator(isotope_pattern_resolution) 
    else:
        isotope_pattern_generator = pyopenms.CoarseIsotopePatternGenerator(isotope_pattern_resolution) 
    
    isotopes = molecular_formula.getIsotopeDistribution(isotope_pattern_generator)
    
    isotopes_aslist = []
    for iso in isotopes.getContainer():
        mass = iso.getMZ()
        intensity = iso.getIntensity()
        isotopes_aslist.append([mass, intensity])
        
    isotopes_asarray = np.asarray(isotopes_aslist)
    return isotopes_asarray   

def fine_grain_isotope_distribution(distribution, peak_width, grid_step_size):
    mass_grid = np.arange(distribution[0, 0], distribution[-1, 0], grid_step_size)
    intensities = np.zeros(mass_grid.shape)
    for peak in distribution:
        intensities += peak[1] * np.exp(-0.5*((mass_grid - peak[0]) / peak_width)**2)
    return mass_grid, intensities



def calculate_abundance_from_ptm_pattern():
    pass



def plot_spectra(data, peaks, gaussian_model, title, unmodified_species_mass=None):
    x = np.arange(data.search_window_start_mass, data.search_window_end_mass)
    y = multi_gaussian(x, gaussian_model.amplitudes, gaussian_model.means, gaussian_model.stddev)
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
        
