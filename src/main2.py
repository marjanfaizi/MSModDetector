#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2021

@author: Marjan Faizi
"""

import glob
from itertools import product
import pandas as pd

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
import utils
import config_sim as config


file_name = [file for file in glob.glob(config.file_names)][0] 
rep = config.replicates[0]
cond = config.conditions[0]

aa_sequence_str = utils.read_fasta(config.fasta_file_name)

mod = Modifications(config.modfication_file_name, aa_sequence_str)
mass_shifts = MassShifts()
data = MassSpecData(file_name)
data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)        

all_peaks = data.picking_peaks()
trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])


noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()

gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)
gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, config.pvalue_threshold)      

mean = gaussian_model.fitting_results["mean"]
amplitude = gaussian_model.fitting_results["amplitude"]

gaussian_model.remove_overlapping_fitting_results()
gaussian_model.refit_amplitudes(trimmed_peaks_in_search_window, noise_level)
gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)

"""
mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, data.rescaling_factor, cond+"_"+rep)


mass_shifts.calculate_avg_mass()

if config.bin_peaks == True:
    mass_shifts.bin_peaks()

if config.calculate_mass_shifts == True:
    mass_shifts.add_mass_shifts()

if config.determine_ptm_patterns == True:
    print("\nSearching for PTM combinations:")

    mass_shifts.determine_ptm_patterns(mod, config.mass_tolerance, config.objective_fun)        
    mass_shifts.add_ptm_patterns_to_table()
    mass_shifts.save_table(mass_shifts.ptm_patterns_df, "../output/ptm_patterns_table.csv")
    
mass_shifts.save_table(mass_shifts.identified_masses_df, "../output/mass_shifts.csv")
    
"""


#####################################################################################################
from matplotlib import pyplot as plt
import numpy as np
modform_distribution =  pd.read_csv('../data/modifications/modform_distribution.csv', sep=',')




masses = modform_distribution["mass"].values+config.unmodified_species_mass
intensities = modform_distribution["relative intensity"].values
x_gauss_func = np.arange(config.mass_start_range, config.mass_end_range)
y_gauss_func = utils.multi_gaussian(x_gauss_func, intensities, masses, config.stddev_isotope_distribution)
y_gauss_func_pred = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results["amplitude"].values*data.rescaling_factor, gaussian_model.fitting_results["mean"].values, config.stddev_isotope_distribution)


plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color="0.3", linewidth=1)
#plt.plot(trimmed_peaks_in_search_window[:,0], trimmed_peaks_in_search_window[:,1]*data.rescaling_factor, 'b.')
plt.plot(mean, amplitude*data.rescaling_factor, 'b.')
plt.plot(gaussian_model.fitting_results["mean"],gaussian_model.fitting_results["amplitude"]*data.rescaling_factor, 'y.')
plt.plot(data.raw_spectrum[:,0],yt_mean*data.rescaling_factor, 'g-')
plt.plot(x_gauss_func, y_gauss_func_pred, 'y-')
plt.plot(modform_distribution["mass"]+config.unmodified_species_mass, modform_distribution["relative intensity"], 'r.')
#plt.plot(x_gauss_func, y_gauss_func, "r-")
plt.axhline(y=noise_level*data.rescaling_factor, c='r', lw=0.3)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43600,44540))
#plt.ylim(ymin=-200)
plt.tight_layout()
plt.show()



from scipy import optimize

def error_func(mean, amplitude, stddev, x, y):
    return utils.multi_gaussian(x, amplitude, mean, stddev) - y 


refitted_means = optimize.least_squares(error_func, bounds=(config.mass_start_range, config.mass_end_range),
                                        x0=gaussian_model.fitting_results["mean"].values, 
                                        args=(gaussian_model.fitting_results["amplitude"]*data.rescaling_factor, gaussian_model.stddev, data.raw_spectrum[:,0], data.raw_spectrum[:,1]))








import GPy

m_full = GPy.models.GPRegression(np.reshape(trimmed_peaks_in_search_window[:,0], (799,1)),np.reshape(trimmed_peaks_in_search_window[:,1], (799,1)))
_ = m_full.optimize()




