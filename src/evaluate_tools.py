#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 5 2022

@author: Marjan Faizi
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np
import itertools

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
from simulate_data import SimulateData
import utils
import config_sim as config

sns.set_color_codes("pastel")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1}                  
sns.set_context("paper", rc = paper_rc)   


###################################################################################################################
################################################### INPUT DATA ####################################################
###################################################################################################################
aa_sequence_str = utils.read_fasta(config.fasta_file_name)
modifications_table =  pd.read_csv(config.modfication_file_name, sep=';')
modform_distribution =  pd.read_csv("../data/modform_distributions/modform_distribution_phospho.csv", sep=",")
error_estimate_table = pd.read_csv("../output/noise_distribution_table.csv")
###################################################################################################################
###################################################################################################################


###################################################################################################################
############################################### REQUIRED FUNCTIONS ################################################
###################################################################################################################
def determine_pred_vectors(detected_mass_shifts, modform_distribution_simulated):
    mass_shift_true = modform_distribution_simulated["mass"].values
    detected_mass_shifts
    mass_shift_pred = []
    ptm_pattern_pred = []
    for ms in mass_shift_true:
        diff = np.abs(detected_mass_shifts["mass shift"].values - ms)
        nearest_mass_index = diff.argmin()
        if diff[nearest_mass_index] <= config.mass_tolerance:
            mass_shift_pred += [detected_mass_shifts.loc[nearest_mass_index, "mass shift"]]
            ptm_pattern_pred += [detected_mass_shifts.loc[nearest_mass_index, "PTM pattern"]]
        else:
           mass_shift_pred += [0] 
    return mass_shift_pred, ptm_pattern_pred
###################################################################################################################
###################################################################################################################


###################################################################################################################
############################################## DETERMINE NOISE LEVEL ##############################################
###################################################################################################################
sigma_mean = error_estimate_table[error_estimate_table["sigma noise"]<0.4]["sigma noise"].mean()
sigma_std = error_estimate_table[error_estimate_table["sigma noise"]<0.4]["sigma noise"].std()

vertical_noise_std_list = [0, 0.0625, 0.125, 0.25]
horizontal_noise_std_list = [0, 0.025, 0.05, 0.1]
sigma_std_list = [0, sigma_std/4, sigma_std/2, sigma_std]

all_std_combinations = [p for p in itertools.product(*[sigma_std_list, horizontal_noise_std_list, 
                                                       vertical_noise_std_list])]
###################################################################################################################
###################################################################################################################


###################################################################################################################
################################ GENERATE SIMULATED DATA AND RUN ALGORITHM ON IT ##################################
###################################################################################################################
mod = Modifications(config.modfication_file_name, aa_sequence_str)

data_simulation = SimulateData(aa_sequence_str, modifications_table)
data_simulation.set_sigma_mean(sigma_mean)

progress_bar_count = 0

score = []
acc = []
for std_comb in all_std_combinations:
    # simulated data
    data_simulation.reset_noise_levels()
    data_simulation.add_noise(vertical_noise_std=std_comb[0], sigma_noise_std=std_comb[1], 
                              horizontal_noise_std=std_comb[2])
    masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)
    
    theoretical_spectrum_file_name = "../output/phospho_with_noise.csv"
    data_simulation.save_mass_spectrum(masses, intensities, theoretical_spectrum_file_name)
    
    # determine mass shifts and ptm patterns
    mass_shifts = MassShifts(config.mass_start_range, config.mass_end_range)
        
    data = MassSpecData(theoretical_spectrum_file_name)
    data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)        
    
    all_peaks = data.picking_peaks()
    trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
    trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
    trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])
    
    noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
    
    gaussian_model = GaussianModel("simulated", config.stddev_isotope_distribution)
    gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)
    gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, config.pvalue_threshold)      
    gaussian_model.remove_overlapping_fitting_results()
    gaussian_model.refit_results(trimmed_peaks_in_search_window, noise_level, refit_mean=True)
    gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)
     
    mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, data.rescaling_factor, "simulated")
    
    mass_shifts.calculate_avg_mass()
    mass_shifts.add_mass_shifts(config.unmodified_species_mass)
    
    mass_shifts.determine_ptm_patterns(mod, config.mass_tolerance, config.objective_fun)        
    mass_shifts.add_ptm_patterns_to_table()
    
    # determine performance of prediction
    mass_shift_true = modform_distribution["mass"].values
    ptm_pattern_true = modform_distribution["modform"].values
    
    mass_shift_pred, ptm_pattern_pred = determine_pred_vectors(mass_shifts.identified_masses_df, modform_distribution)
    
    score += [r2_score(mass_shift_true, mass_shift_pred)]
    acc += [len(set(ptm_pattern_true) & set(ptm_pattern_pred)) / len(ptm_pattern_true)]
    
    progress_bar_count += 1        
    utils.progress(progress_bar_count, len(all_std_combinations))
###################################################################################################################
###################################################################################################################



data_simulation.reset_noise_levels()
data_simulation.add_noise(vertical_noise_std=0.25, sigma_noise_std=0, 
                          basal_noise_beta=30, horizontal_noise_std=0)
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

plt.figure(figsize=(7,3))
plt.plot(masses, intensities, "b.-")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43700, 44200))
plt.ylim(ymin=-0.005)
plt.tight_layout()
sns.despine()
plt.show()



