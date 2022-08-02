#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 5 2022

@author: Marjan Faizi
"""

import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import itertools
from scipy import stats
#from scipy.optimize import minimize

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
from simulate_data import SimulateData
import utils
import config_sim as config 


###################################################################################################################
################################################### INPUT DATA ####################################################
###################################################################################################################
modform_file_name = "complex"
protein_entries = utils.read_fasta(config.fasta_file_name)
aa_sequence_str = list(protein_entries.values())[0]
modifications_table = pd.read_csv(config.modfication_file_name, sep=";")
modform_distribution = pd.read_csv("../data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", 
                                   sep=",")
modform_distribution["rel. intensity"] = modform_distribution["intensity"]/ modform_distribution["intensity"].sum()
error_estimate_table = pd.read_csv("../output/error_noise_distribution_table.csv")
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
basal_noise = error_estimate_table["basal noise (a.u.)"].values
horizontal_error = error_estimate_table[(error_estimate_table["horizontal error (Da)"]<0.3) &
                                        (error_estimate_table["horizontal error (Da)"]>-0.3) &
                                        (error_estimate_table["is_signal"]==True)]["horizontal error (Da)"].values
vertical_error = error_estimate_table[(error_estimate_table["is_signal"]==True)]["vertical error (rel.)"].values

vertical_error_par = list(stats.beta.fit(vertical_error))
horizontal_error_par = list(stats.beta.fit(horizontal_error)) 
basal_noise_par = list(stats.beta.fit(basal_noise))
###################################################################################################################
###################################################################################################################


###################################################################################################################
################################ GENERATE SIMULATED DATA AND RUN ALGORITHM ON IT ##################################
###################################################################################################################
def variable_error_noise_performance(data_simulation, mod, modform_distribution, repeat_simulation, 
                                     vertical_error_par, horizontal_error_par, basal_noise_par,
                                     gaussian_method="one"):

    error_noise_combinations = list(itertools.product([0, 1], repeat=3))#4))
    
    performance_df = pd.DataFrame(columns=["vertical_error", "horizontal_error", "basal_noise", 
                                           "all_detected_mass_shifts", "simulated_mass_shifts", 
                                           "matching_mass_shifts", "mass_shift_deviation", "r_score_abundance", 
                                           "matching_ptm_patterns_min_ptm", "matching_ptm_patterns_min_error",
                                           "matching_ptm_patterns_min_both"])
    
    performance_df["vertical_error"] = [a_tuple[0] for a_tuple in error_noise_combinations]
    performance_df["horizontal_error"] = [a_tuple[1] for a_tuple in error_noise_combinations]
    performance_df["basal_noise"] = [a_tuple[2] for a_tuple in error_noise_combinations]
    performance_df["simulated_mass_shifts"] = modform_distribution.shape[0]
    
    objective_func = ["min_err", "min_ptm", "min_both"]
    
    mass_shift_deviation_avg = []
    all_detected_mass_shifts_avg = []
    matching_mass_shifts_avg = []
    matching_ptm_patterns_min_err_avg = []
    matching_ptm_patterns_min_ptm_avg = []
    matching_ptm_patterns_min_both_avg = []
    r_score_abundance_avg = []
    
    progress = 1

    for comb in error_noise_combinations:
        mass_shift_deviation = []
        all_detected_mass_shifts = []
        matching_mass_shifts = []
        r_score_mass = []
        r_score_abundance = []
        matching_ptm_patterns_min_err = []
        matching_ptm_patterns_min_ptm = []
        matching_ptm_patterns_min_both = []
        
        for repeat in range(repeat_simulation):
            # simulated data
            data_simulation.reset_noise_error()
            if comb[0]: vertical_error = vertical_error_par
            else: vertical_error = None
            if comb[1]: horizontal_error = horizontal_error_par
            else: horizontal_error = None
            if comb[2]: basal_noise = basal_noise_par
            else: basal_noise = None
                    
            data_simulation.add_noise(vertical_error_par=vertical_error, horizontal_error_par=horizontal_error,
                                      basal_noise_par=basal_noise)
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
            trimmed_peaks_in_search_window_above_noise = trimmed_peaks_in_search_window[trimmed_peaks_in_search_window[:,1]>noise_level]
        
            gaussian_model = GaussianModel("simulated", config.stddev_isotope_distribution)
            gaussian_model.determine_variable_window_sizes(config.unmodified_species_mass, config.window_size_lb, config.window_size_ub)
            
            if gaussian_method == "one":
                gaussian_model.fit_gaussian_within_window(trimmed_peaks_in_search_window_above_noise, config.allowed_overlap_fitting_window, config.pvalue_threshold, noise_level)      
            elif gaussian_method == "two":
                gaussian_model.fit_two_gaussian_within_window(trimmed_peaks_in_search_window_above_noise, config.pvalue_threshold, noise_level)      
            
            gaussian_model.refit_results(trimmed_peaks_in_search_window_above_noise, noise_level, refit_mean=True)
            gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)
        
            mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, "simulated") # data.rescaling_factor
    
            if mass_shifts.identified_masses_df.empty:
                all_detected_mass_shifts += [0]
                matching_mass_shifts += [0]
                matching_ptm_patterns_min_err += [0]
                matching_ptm_patterns_min_ptm += [0]
                matching_ptm_patterns_min_both += [0]
                r_score_mass += [np.nan]
                r_score_abundance += [np.nan]
                    
            else:
                mass_shifts.calculate_avg_mass()
                mass_shifts.add_mass_shifts(config.unmodified_species_mass)
                
                mass_shift_pred_ix, mass_shift_true_ix = determine_matching_mass_shifts(mass_shifts.identified_masses_df, 
                                                                                        modform_distribution)
                
                mass_shift_true = modform_distribution.loc[mass_shift_true_ix, "mass"].values
                abundance_true = modform_distribution.loc[mass_shift_true_ix, "rel. intensity"].values
                mass_shift_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "mass shift"].values
               
                abundance_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "rel. abundances simulated"].values
            
                mass_shift_deviation += [np.mean(abs(mass_shift_true-mass_shift_pred)/config.mass_tolerance)]
                all_detected_mass_shifts += [mass_shifts.identified_masses_df.shape[0]]
                matching_mass_shifts += [len(mass_shift_pred)]
                
                if len(mass_shift_true)>1:
                    r_score_abundance += [r2_score(abundance_true, abundance_pred)]
                else:
                    r_score_abundance += [np.nan]

                for func in objective_func:
                    mass_shifts.determine_ptm_patterns(mod, config.mass_tolerance, func, config.laps_run_lp,  msg_progress=False)
                    mass_shifts.add_ptm_patterns_to_table()
                    ptm_pattern_true = modform_distribution.loc[mass_shift_true_ix, "PTM pattern"].values
                    ptm_pattern_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "PTM pattern"].values
                    if func == "min_err":
                        matching_ptm_patterns_min_err += [len(set(ptm_pattern_true) & set(ptm_pattern_pred))]
                    elif func == "min_ptm":
                        matching_ptm_patterns_min_ptm += [len(set(ptm_pattern_true) & set(ptm_pattern_pred))]
                    elif func == "min_both":
                        matching_ptm_patterns_min_both += [len(set(ptm_pattern_true) & set(ptm_pattern_pred))]
                
        mass_shift_deviation_avg += [np.mean(np.array(mass_shift_deviation))]
        all_detected_mass_shifts_avg += [np.mean(np.array(all_detected_mass_shifts))]
        matching_mass_shifts_avg += [np.mean(np.array(matching_mass_shifts))]
        matching_ptm_patterns_min_err_avg += [np.mean(np.array(matching_ptm_patterns_min_err))]
        matching_ptm_patterns_min_ptm_avg += [np.mean(np.array(matching_ptm_patterns_min_ptm))]
        matching_ptm_patterns_min_both_avg += [np.mean(np.array(matching_ptm_patterns_min_both))]
        r_score_abundance_avg += [np.nanmean(np.array(r_score_abundance))]
        
        print(progress, "out of", len(error_noise_combinations))
        progress += 1
    
    performance_df["all_detected_mass_shifts"] = all_detected_mass_shifts_avg
    performance_df["mass_shift_deviation"] = mass_shift_deviation_avg
    performance_df["matching_mass_shifts"] = matching_mass_shifts_avg
    performance_df["r_score_abundance"] = r_score_abundance_avg
    performance_df["matching_ptm_patterns_min_err"] = matching_ptm_patterns_min_err_avg
    performance_df["matching_ptm_patterns_min_ptm"] = matching_ptm_patterns_min_ptm_avg
    performance_df["matching_ptm_patterns_min_both"] = matching_ptm_patterns_min_both_avg
    
    return performance_df
###################################################################################################################
###################################################################################################################


###################################################################################################################
################################## TEST PERFORMANCE ON OVERLAPPING DISTRIBUTION ###################################
###################################################################################################################
def test_overlapping_mass_detection(data_simulation, vertical_error, horizontal_error, basal_noise,
                                    modform_distribution, repeat_simulation, gaussian_method="one"):    
    
    mass_shift = np.full([repeat_simulation, modform_distribution.shape[0]], 0)
    p_values = np.full([repeat_simulation, modform_distribution.shape[0]], np.nan)
    chi_sqaure_score = p_values.copy(); mass_shift_deviation = p_values.copy(); ptm_patterns = p_values.copy()
    ptm_patterns_top3 = p_values.copy(); ptm_patterns_top5 = p_values.copy(); ptm_patterns_top10 = p_values.copy()

    objective_func = "min_ptm"

    for repeat in range(repeat_simulation):
        data_simulation.add_noise(vertical_error_par=vertical_error, horizontal_error_par=horizontal_error,
                                  basal_noise_par=basal_noise)
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
        trimmed_peaks_in_search_window_above_noise = trimmed_peaks_in_search_window[trimmed_peaks_in_search_window[:,1]>noise_level]
    
        gaussian_model = GaussianModel("simulated", config.stddev_isotope_distribution)
        gaussian_model.determine_variable_window_sizes(config.unmodified_species_mass, config.window_size_lb, config.window_size_ub)
    
        if gaussian_method == "one":
            gaussian_model.fit_gaussian_within_window(trimmed_peaks_in_search_window_above_noise, config.allowed_overlap_fitting_window, config.pvalue_threshold, noise_level)      
        elif gaussian_method == "two":
            gaussian_model.fit_two_gaussian_within_window(trimmed_peaks_in_search_window_above_noise, config.pvalue_threshold, noise_level)      
        
        gaussian_model.refit_results(trimmed_peaks_in_search_window_above_noise, noise_level, refit_mean=True)
        gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)

        mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, "simulated") 
        
        if not mass_shifts.identified_masses_df.empty:
            mass_shifts.calculate_avg_mass()
            mass_shifts.add_mass_shifts(config.unmodified_species_mass)
            mass_shifts.determine_ptm_patterns(mod, config.mass_tolerance, objective_func, config.laps_run_lp, msg_progress=False)
            mass_shifts.add_ptm_patterns_to_table()
            ## find matching mass shhifts and store their scores, pvalues and ptm pattern
            mass_shift_pred_ix, mass_shift_true_ix = determine_matching_mass_shifts(mass_shifts.identified_masses_df, 
                                                                                    modform_distribution)
           
            mass_shift_true = modform_distribution.loc[mass_shift_true_ix, "mass"].values
            mass_shift_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "mass shift"].values
            mass_shift[repeat, mass_shift_true_ix] = 1
            p_values[repeat, mass_shift_true_ix] = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "pvalue simulated"].values
            chi_sqaure_score[repeat, mass_shift_true_ix] = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "chi_score simulated"].values
            mass_shift_deviation[repeat, mass_shift_true_ix] = abs(mass_shift_true-mass_shift_pred)/config.mass_tolerance                     

            for ix in range(len(mass_shift_true_ix)):      
                
                true_ptm_pattern = modform_distribution.loc[mass_shift_true_ix[ix], "PTM pattern"] 
                pred_ptm_list = mass_shifts.ptm_patterns_df[mass_shifts.identified_masses_df.loc[mass_shift_pred_ix[ix], "mass shift"]==mass_shifts.ptm_patterns_df["mass shift"]]["PTM pattern"].values

                if modform_distribution.loc[mass_shift_true_ix[ix], "PTM pattern"] == mass_shifts.identified_masses_df.loc[mass_shift_pred_ix[ix], "PTM pattern"]:
                    ptm_patterns[repeat, mass_shift_true_ix[ix]] = 1
                else: 
                    ptm_patterns[repeat, mass_shift_true_ix[ix]] = 0
                # is ptm prediction in the top 3
                if true_ptm_pattern in pred_ptm_list[:3] or modform_distribution.loc[mass_shift_true_ix[ix], "PTM pattern"] == mass_shifts.identified_masses_df.loc[mass_shift_pred_ix[ix], "PTM pattern"]:
                    ptm_patterns_top3[repeat, mass_shift_true_ix[ix]] = 1
                else:
                    ptm_patterns_top3[repeat, mass_shift_true_ix[ix]] = 0
                # is ptm prediction in the top 5
                if true_ptm_pattern in pred_ptm_list[:5] or modform_distribution.loc[mass_shift_true_ix[ix], "PTM pattern"] == mass_shifts.identified_masses_df.loc[mass_shift_pred_ix[ix], "PTM pattern"]:
                    ptm_patterns_top5[repeat, mass_shift_true_ix[ix]] = 1
                else:
                    ptm_patterns_top5[repeat, mass_shift_true_ix[ix]] = 0
                # is ptm prediction in the top 10
                if true_ptm_pattern in pred_ptm_list[:5] or modform_distribution.loc[mass_shift_true_ix[ix], "PTM pattern"] == mass_shifts.identified_masses_df.loc[mass_shift_pred_ix[ix], "PTM pattern"]:
                    ptm_patterns_top10[repeat, mass_shift_true_ix[ix]] = 1
                else:
                    ptm_patterns_top10[repeat, mass_shift_true_ix[ix]] = 0                
    
    return  mass_shift, chi_sqaure_score, mass_shift_deviation, ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10
###################################################################################################################
###################################################################################################################

if __name__ == "__main__":
    mod = Modifications(config.modfication_file_name, aa_sequence_str)
    repeat_simulation = 5
    peak_width_mode = 0.25
    data_simulation = SimulateData(aa_sequence_str, modifications_table)
    data_simulation.set_peak_width_mode(peak_width_mode)
    """
    performance_df = variable_error_noise_performance(data_simulation, mod, modform_distribution, repeat_simulation,
                                                      vertical_error_par, horizontal_error_par, basal_noise_par,
                                                      gaussian_method="one")
    
    performance_df.to_csv("../output/performance_"+modform_file_name+".csv", sep=",", index=False) 
    
    """
    mass_shift, chi_sqaure_score, mass_shift_deviation, ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10 = test_overlapping_mass_detection(
                                                                                    data_simulation, 
                                                                                    vertical_error_par, 
                                                                                    horizontal_error_par, 
                                                                                    basal_noise_par, 
                                                                                    modform_distribution, 
                                                                                    repeat_simulation,
                                                                                    gaussian_method="one")

    np.savez("../output/arrays_no_overlap_allowance", mass_shift, chi_sqaure_score, mass_shift_deviation, 
             ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10)
    
    



"""
###################################################################################################################
############################################ CREATE SIMULATED SPECTRUM ############################################
###################################################################################################################
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from simulate_data import SimulateData
import config_sim as config 
import matplotlib.pyplot as plt
import seaborn as sns

mod = Modifications(config.modfication_file_name, aa_sequence_str)
peak_width_mode = 0.25

data_simulation = SimulateData(aa_sequence_str, modifications_table)
data_simulation.set_peak_width_mode(peak_width_mode)

data_simulation.reset_noise_error()
data_simulation.add_noise(vertical_error_par=vertical_error_par, horizontal_error_par=horizontal_error_par,
                          basal_noise_par=basal_noise_par)#, peak_width_par=peak_width_par)

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
trimmed_peaks_in_search_window_above_noise = trimmed_peaks_in_search_window[trimmed_peaks_in_search_window[:,1]>noise_level]

gaussian_model = GaussianModel("simulated", config.stddev_isotope_distribution)
gaussian_model.determine_variable_window_sizes(config.unmodified_species_mass, config.window_size_lb, config.window_size_ub)
gaussian_model.fit_gaussian_within_window(trimmed_peaks_in_search_window, config.allowed_overlap_fitting_window, config.pvalue_threshold, noise_level)      
#gaussian_model.fit_two_gaussian_within_window(trimmed_peaks_in_search_window_above_noise, config.pvalue_threshold, noise_level)      

gaussian_model.refit_results(trimmed_peaks_in_search_window, noise_level, refit_mean=True)
gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)

mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, "simulated") 

mass_shifts.calculate_avg_mass()
mass_shifts.add_mass_shifts(config.unmodified_species_mass)

mass_shifts.determine_ptm_patterns(mod, config.mass_tolerance, config.objective_fun, config.laps_run_lp)     
  
mass_shifts.add_ptm_patterns_to_table()

fig = plt.figure(figsize=(7, 2.5))
plt.plot(data.masses, data.intensities, '-', color="0.3")
plt.plot(modform_distribution["mass"]+config.unmodified_species_mass, modform_distribution["intensity"], '.', markersize=3, color="r")
plt.plot(gaussian_model.fitting_results["mean"], gaussian_model.fitting_results["amplitude"]*data.rescaling_factor, '.', color="b")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43625, 44400))
sns.despine()
fig.tight_layout()
plt.show()

mass_shifts.identified_masses_df

fig = plt.figure(figsize=(14,3.5))
gs = fig.add_gridspec(2, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)
axes[0].plot(masses, intensities, color="0.3") 
axes[1].plot(masses_noise, intensities_noise, color="0.3") 
axes[0].plot(modform_distribution["mass"]+config.unmodified_species_mass, modform_distribution["intensity"], 'o', color="b")
axes[1].plot(modform_distribution["mass"]+config.unmodified_species_mass, modform_distribution["intensity"], 'o', color="b")
axes[0].plot(mean, amplitude, 'o', color="r")
axes[1].plot(mean_noise, amplitude_noise, 'o', color="r")
axes[1].set_xlabel("mass (Da)")#, fontsize=10)
axes[1].set_ylabel("intensity (a.u.)")#, fontsize=10)
[axes[i].set_xlim([43615, 44180]) for i in range(2)] # phospho
[axes[i].set_ylim([0, 1410]) for i in range(2)] # phospho
fig.tight_layout()
sns.despine()
plt.show()



x_gauss_func = np.arange(config.mass_start_range, config.mass_end_range)
y_gauss_func = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results["amplitude"],  
                                    gaussian_model.fitting_results["mean"], config.stddev_isotope_distribution)



#proton_mass = 1.007
#masses += proton_mass
#data_simulation.save_mass_spectrum(masses, intensities, 
#                                   "../output/ptm_pattern_"+modform_file_name+"_with_error_noise.csv")




plt.figure(figsize=(6, 3))
plt.plot(data.masses, data.intensities/data.rescaling_factor, '-', color="0.3")
plt.plot(x_gauss_func, utils.gaussian(x_gauss_func, gaussian_model.fitting_results.loc[0,"amplitude"], 
                                      gaussian_model.fitting_results.loc[0,"mean"], config.stddev_isotope_distribution), 
        '-', color="red")
plt.axhline(y=noise_level, color='r', linestyle='-')
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("intensity (a.u.)", fontsize=10)
plt.tight_layout()
plt.show()






import numpy as np
from cvxopt import matrix, glpk
from modifications import Modifications

import utils
import config_sim as config 

protein_entries = utils.read_fasta(config.fasta_file_name)
aa_sequence_str = list(protein_entries.values())[0]
mod = Modifications(config.modfication_file_name, aa_sequence_str)

observed_mass_shift = 1
max_mass_error = config.mass_tolerance
min_number_ptms = 0



number_variables = len(mod.ptm_masses)
ones = np.ones(number_variables)
inequality_lhs = np.vstack([np.array(mod.ptm_masses), -np.array(mod.ptm_masses), -np.identity(number_variables), np.identity(number_variables), -ones])
A = matrix(inequality_lhs)
lower_bounds = np.zeros(number_variables)
inequality_rhs = np.vstack([observed_mass_shift+max_mass_error, -observed_mass_shift+max_mass_error, lower_bounds.reshape(-1,1), 
                            np.array(mod.upper_bounds).reshape(-1,1), -min_number_ptms])
b = matrix(inequality_rhs)
c = matrix(ones)
status, solution = glpk.ilp(c, A, b, I=set(range(number_variables)))


###################################################################################################################
###################################################################################################################
"""
