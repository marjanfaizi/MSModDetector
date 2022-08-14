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


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5*((x - mean) / stddev)**2)
    

def integral_of_gaussian(amplitude, stddev):
    return np.sqrt(2*np.pi)*amplitude*stddev

 
def multi_gaussian(x, amplitude, mean, stddev):
    y = np.zeros(len(x))
    for i in range(len(mean)):
        y += gaussian(x, amplitude[i], mean[i], stddev)

    return y  


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


