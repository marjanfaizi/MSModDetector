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
modform_file_name = "phospho"
aa_sequence_str = utils.read_fasta(config.fasta_file_name)
modifications_table = pd.read_csv(config.modfication_file_name, sep=';')
modform_distribution = pd.read_csv("../data/modform_distributions/modform_distribution_"+modform_file_name+".csv", 
                                   sep=",")
error_estimate_table = pd.read_csv("../output/noise_distribution_table.csv")
###################################################################################################################
###################################################################################################################


###################################################################################################################
############################################### REQUIRED FUNCTIONS ################################################
###################################################################################################################
def determine_matching_mass_shifts(detected_mass_shifts, modform_distribution_simulated):
    mass_shift_true = modform_distribution_simulated["mass"].values
    detected_mass_shifts
    mass_shift_pred_ix = []
    mass_shift_true_ix = []
    for ms_true_index, ms_true in enumerate(mass_shift_true):
        diff = np.abs(detected_mass_shifts["mass shift"].values - ms_true)
        nearest_predicted_mass_index = diff.argmin()
        if diff[nearest_predicted_mass_index] <= config.mass_tolerance:
            mass_shift_pred_ix += [nearest_predicted_mass_index]
            mass_shift_true_ix += [ms_true_index]
    return mass_shift_pred_ix, mass_shift_true_ix
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
basal_noise_beta_list = [0, 1/400, 1/200, 1/100]

all_std_combinations = [p for p in itertools.product(*[sigma_std_list, horizontal_noise_std_list, 
                                                       vertical_noise_std_list, basal_noise_beta_list])]
###################################################################################################################
###################################################################################################################


###################################################################################################################
################################ GENERATE SIMULATED DATA AND RUN ALGORITHM ON IT ##################################
###################################################################################################################
mod = Modifications(config.modfication_file_name, aa_sequence_str)

data_simulation = SimulateData(aa_sequence_str, modifications_table)
data_simulation.set_sigma_mean(sigma_mean)

performance_df = pd.DataFrame(columns=["vertical_noise_std", "sigma_noise_std", "horizontal_noise_std", 
                                       "basal_noise_beta", "all_detected_mass_shifts", "simulated_mass_shifts", 
                                       "matching_mass_shifts", "r_score", "matching_ptm_patterns"])

performance_df["sigma_noise_std"] = [a_tuple[0] for a_tuple in all_std_combinations]
performance_df["horizontal_noise_std"] = [a_tuple[1] for a_tuple in all_std_combinations]
performance_df["vertical_noise_std"] = [a_tuple[2] for a_tuple in all_std_combinations]
performance_df["basal_noise_beta"] = [a_tuple[3] for a_tuple in all_std_combinations]
performance_df["simulated_mass_shifts"] = modform_distribution.shape[0]

all_detected_mass_shifts = []
matching_mass_shifts = []
r_score = []
matching_ptm_patterns = []
progress = 1
for std_comb in all_std_combinations:
    # simulated data
    data_simulation.reset_noise_levels()
    data_simulation.add_noise(vertical_noise_std=std_comb[0], sigma_noise_std=std_comb[1], 
                              horizontal_noise_std=std_comb[2], basal_noise_beta=std_comb[3])
    masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)
    
    theoretical_spectrum_file_name = "../output/spectrum_"+modform_file_name+".csv"
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
    mass_shift_pred_ix, mass_shift_true_ix = determine_matching_mass_shifts(mass_shifts.identified_masses_df, 
                                                                            modform_distribution)
    
    mass_shift_true = modform_distribution.loc[mass_shift_true_ix, "mass"].values
    ptm_pattern_true = modform_distribution.loc[mass_shift_true_ix, "modform"].values
    mass_shift_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "mass shift"].values
    ptm_pattern_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "PTM pattern"].values
    
    all_detected_mass_shifts += [mass_shifts.identified_masses_df.shape[0]]
    matching_mass_shifts += [len(mass_shift_pred)]
    r_score += [r2_score(mass_shift_true, mass_shift_pred)]
    matching_ptm_patterns += [len(set(ptm_pattern_true) & set(ptm_pattern_pred))]

    print(progress, "out of", len(all_std_combinations))
    progress += 1


performance_df["all_detected_mass_shifts"] = all_detected_mass_shifts
performance_df["matching_mass_shifts"] = matching_mass_shifts
performance_df["r_score"] = r_score
performance_df["matching_ptm_patterns"] = matching_ptm_patterns
performance_df.to_csv("../output/performance_"+modform_file_name+".csv", sep=",", index=False) 
###################################################################################################################
###################################################################################################################



###################################################################################################################
###################################################################################################################

basal_noise_sigma_comb = [p for p in itertools.product(*[basal_noise_beta_list[::-1], sigma_std_list])]

performance_df["ptm_pattern_acc"]=performance_df["matching_ptm_patterns"]/performance_df["simulated_mass_shifts"]

metric = "r_score" # "ptm_pattern_acc" 

fig, axn = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(7,6))
cbar_ax = fig.add_axes([.93, 0.3, 0.02, 0.4])
cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
for i, ax in enumerate(axn.flat):
    basal_noise_beta = basal_noise_sigma_comb[i][0]
    sigma_std = round(basal_noise_sigma_comb[i][1],6)
    mask = ((performance_df["sigma_noise_std"].round(6)==sigma_std) & 
            (performance_df["basal_noise_beta"]==basal_noise_beta))
    pivot_df = performance_df[mask].pivot("horizontal_noise_std", "vertical_noise_std", metric)                                                                             
    sns.heatmap(pivot_df, ax=ax,
                cbar=i == 0, cmap=cmap,
                vmin=performance_df[metric].min(), vmax=performance_df[metric].max(),
                cbar_ax=None if i else cbar_ax)
    
    ax.set_xlabel("$\sigma_{vertical\_noise}$")
    ax.set_ylabel("$\sigma_{horizontal\_noise}$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)    
    ax.invert_yaxis()
    matching_mass_shifts = performance_df[mask]["matching_mass_shifts"].values[i]
    simulated_mass_shifts = performance_df[mask]["simulated_mass_shifts"].values[i]
    ax.set_title(str(matching_mass_shifts)+" out of "+str(simulated_mass_shifts))
       
fig.tight_layout(rect=[0, 0, 0.93, 1])






"""
data_simulation.reset_noise_levels()
data_simulation.add_noise(vertical_noise_std=0.125, sigma_noise_std=sigma_std/2, basal_noise_beta=1/200, 
                          horizontal_noise_std=0.05)
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
"""
###################################################################################################################
###################################################################################################################


