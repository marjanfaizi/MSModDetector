#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 5 2022

@author: Marjan Faizi
"""

import sys
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import itertools
from scipy import stats

sys.path.append("..")
sys.path.append("../../")

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
import utils
from simulate_data import SimulateData
import config 


###################################################################################################################
############################################### REQUIRED FUNCTIONS ################################################
###################################################################################################################
def determine_matching_mass_shifts(detected_mass_shifts, modform_distribution_simulated, mass_tolerance):
    mass_shift_true = modform_distribution_simulated["mass"].values
    detected_mass_shifts
    mass_shift_pred_ix = []
    mass_shift_true_ix = []
    for ms_true_index, ms_true in enumerate(mass_shift_true):
        diff = np.abs(detected_mass_shifts["mass shift"].values - ms_true)
        nearest_predicted_mass_index = diff.argmin()
        if diff[nearest_predicted_mass_index] <= mass_tolerance:
            mass_shift_pred_ix += [nearest_predicted_mass_index]
            mass_shift_true_ix += [ms_true_index]
    return mass_shift_pred_ix, mass_shift_true_ix
###################################################################################################################
###################################################################################################################


###################################################################################################################
################################ GENERATE SIMULATED DATA AND RUN ALGORITHM ON IT ##################################
###################################################################################################################
def variable_error_noise_performance(data_simulation, mod, modform_distribution, repeat_simulation, 
                                     vertical_error_par, horizontal_error_par, basal_noise_par):

    protein_entries = utils.read_fasta("../../"+config.fasta_file_name)
    protein_sequence = list(protein_entries.values())[0]
    unmodified_species_mass, stddev_isotope_distribution = utils.isotope_distribution_fit_par(protein_sequence, 100)
    mass_tolerance = config.mass_error_ppm*1e-6*unmodified_species_mass
    
    
    error_noise_combinations = list(itertools.product([0, 1], repeat=3))
    
    performance_df = pd.DataFrame(columns=["vertical_error", "horizontal_error", "basal_noise", 
                                           "all_detected_mass_shifts", "simulated_mass_shifts", 
                                           "matching_mass_shifts", "mass_shift_deviation", "r_score_abundance", 
                                           "matching_ptm_patterns_min_ptm", "matching_ptm_patterns_min_err",
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
            spectrum = np.column_stack((masses, intensities))
                
            data = MassSpecData(spectrum)
            data.set_mass_range_of_interest(data.raw_spectrum[0,0], data.raw_spectrum[-1,0])
            all_peaks = data.picking_peaks()
            peaks_normalized = data.preprocess_peaks(all_peaks, config.distance_threshold_adjacent_peaks)
            noise_level = config.noise_level_fraction*peaks_normalized[:,1].std()

            gaussian_model = GaussianModel("simulated", stddev_isotope_distribution, config.window_size)
            gaussian_model.fit_gaussian_within_window(peaks_normalized, noise_level, config.pvalue_threshold, config.allowed_overlap)      
            gaussian_model.refit_results(peaks_normalized, noise_level, refit_mean=True)
            gaussian_model.calculate_relative_abundaces(data.search_window_start, data.search_window_end)

            mass_shifts = MassShifts(data.raw_spectrum[0,0], data.raw_spectrum[-1,0])
            mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results,  "simulated")
 
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
                mass_shifts.add_mass_shifts(unmodified_species_mass)
                
                mass_shift_pred_ix, mass_shift_true_ix = determine_matching_mass_shifts(mass_shifts.identified_masses_df, 
                                                                                        modform_distribution, mass_tolerance)
                
                mass_shift_true = modform_distribution.loc[mass_shift_true_ix, "mass"].values
                abundance_true = modform_distribution.loc[mass_shift_true_ix, "rel. intensity"].values
                mass_shift_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "mass shift"].values
               
                abundance_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "rel. abundances simulated"].values
            
                mass_shift_deviation += [np.mean(abs(mass_shift_true-mass_shift_pred)/mass_tolerance)]
                all_detected_mass_shifts += [mass_shifts.identified_masses_df.shape[0]]
                matching_mass_shifts += [len(mass_shift_pred)]
                
                if len(mass_shift_true)>1:
                    r_score_abundance += [r2_score(abundance_true, abundance_pred)]
                else:
                    r_score_abundance += [np.nan]

                for func in objective_func:
                    mass_shifts.determine_ptm_patterns(mod, mass_tolerance, func, config.laps_run_lp,  msg_progress=False)
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
                                    modform_distribution, repeat_simulation, objective_func):    
    
    protein_entries = utils.read_fasta("../../"+config.fasta_file_name)
    protein_sequence = list(protein_entries.values())[0]
    unmodified_species_mass, stddev_isotope_distribution = utils.isotope_distribution_fit_par(protein_sequence, 100)
    mass_tolerance = config.mass_error_ppm*1e-6*unmodified_species_mass
    
    mass_shift = np.full([repeat_simulation, modform_distribution.shape[0]], 0)
    p_values = np.full([repeat_simulation, modform_distribution.shape[0]], np.nan)
    chi_sqaure_score = p_values.copy(); mass_shift_deviation = p_values.copy(); ptm_patterns = p_values.copy()
    ptm_patterns_top3 = p_values.copy(); ptm_patterns_top5 = p_values.copy(); ptm_patterns_top10 = p_values.copy()

    for repeat in range(repeat_simulation):
        data_simulation.add_noise(vertical_error_par=vertical_error, horizontal_error_par=horizontal_error,
                                  basal_noise_par=basal_noise)
        masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)
        spectrum = np.column_stack((masses, intensities))
                
        data = MassSpecData(spectrum)
        data.set_mass_range_of_interest(data.raw_spectrum[0,0], data.raw_spectrum[-1,0])
        all_peaks = data.picking_peaks()
        peaks_normalized = data.preprocess_peaks(all_peaks, config.distance_threshold_adjacent_peaks)
        noise_level = config.noise_level_fraction*peaks_normalized[:,1].std()

        gaussian_model = GaussianModel("simulated", stddev_isotope_distribution, config.window_size)
        gaussian_model.fit_gaussian_within_window(peaks_normalized, noise_level, config.pvalue_threshold, config.allowed_overlap)      
        gaussian_model.refit_results(peaks_normalized, noise_level, refit_mean=True)
        gaussian_model.calculate_relative_abundaces(data.search_window_start, data.search_window_end)

        mass_shifts = MassShifts(data.raw_spectrum[0,0], data.raw_spectrum[-1,0])
        mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results,  "simulated") 
        
        if not mass_shifts.identified_masses_df.empty:
            mass_shifts.calculate_avg_mass()
            mass_shifts.add_mass_shifts(unmodified_species_mass)
            mass_shifts.determine_ptm_patterns(mod, mass_tolerance, objective_func, config.laps_run_lp, msg_progress=False)
            mass_shifts.add_ptm_patterns_to_table()
            ## find matching mass shhifts and store their scores, pvalues and ptm pattern
            mass_shift_pred_ix, mass_shift_true_ix = determine_matching_mass_shifts(mass_shifts.identified_masses_df, 
                                                                                    modform_distribution, mass_tolerance)
           
            mass_shift_true = modform_distribution.loc[mass_shift_true_ix, "mass"].values
            mass_shift_pred = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "mass shift"].values
            mass_shift[repeat, mass_shift_true_ix] = 1
            p_values[repeat, mass_shift_true_ix] = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "pvalue simulated"].values
            chi_sqaure_score[repeat, mass_shift_true_ix] = mass_shifts.identified_masses_df.loc[mass_shift_pred_ix, "chi_score simulated"].values
            mass_shift_deviation[repeat, mass_shift_true_ix] = abs(mass_shift_true-mass_shift_pred)/mass_tolerance                     

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
                if true_ptm_pattern in pred_ptm_list[:10] or modform_distribution.loc[mass_shift_true_ix[ix], "PTM pattern"] == mass_shifts.identified_masses_df.loc[mass_shift_pred_ix[ix], "PTM pattern"]:
                    ptm_patterns_top10[repeat, mass_shift_true_ix[ix]] = 1
                else:
                    ptm_patterns_top10[repeat, mass_shift_true_ix[ix]] = 0                
    
    return  mass_shift, chi_sqaure_score, mass_shift_deviation, ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10
###################################################################################################################
###################################################################################################################

if __name__ == "__main__":
    
    ### specify how many simulation should be run
    repeat_simulation = 2
    
    ### estimated error and noise 
    error_estimate_table = pd.read_csv("../../output/error_noise_distribution_table.csv")
    basal_noise = error_estimate_table["basal_noise"].values
    horizontal_error = error_estimate_table[(error_estimate_table["horizontal_error"]<0.3) &
                                            (error_estimate_table["horizontal_error"]>-0.3) &
                                            (error_estimate_table["is_signal"]==True)]["horizontal_error"].values
    vertical_error = error_estimate_table[(error_estimate_table["is_signal"]==True)]["vertical_error"].values
    vertical_error_par = list(stats.beta.fit(vertical_error))
    horizontal_error_par = list(stats.beta.fit(horizontal_error)) 
    basal_noise_par = list(stats.beta.fit(basal_noise))    
    
    ### PTM database
    modifications_table = pd.read_csv("../../"+config.modfication_file_name, sep=";")
    modifications_table["unimod_id"] = modifications_table["unimod_id"].astype("Int64")
    
    ### create instance of the SimulateData class
    protein_entries = utils.read_fasta("../../"+config.fasta_file_name)
    protein_sequence = list(protein_entries.values())[0]
    mod = Modifications("../../"+config.modfication_file_name, protein_sequence)
    data_simulation = SimulateData(protein_sequence, modifications_table)
    
    
    ### simulated overlapping isotopic ditributions
    modform_file_name = "overlap"
    modform_distribution = pd.read_csv("ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")
    modform_distribution["rel. intensity"] = modform_distribution["intensity"]/ modform_distribution["intensity"].sum()
    
    ### evaluate algorithm on overlapping isotopic ditribution data, how much overlap can be tolerated?
    mass_shift, chi_sqaure_score, mass_shift_deviation, ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10 = test_overlapping_mass_detection(
                                                                                    data_simulation, 
                                                                                    vertical_error_par, 
                                                                                    horizontal_error_par, 
                                                                                    basal_noise_par, 
                                                                                    modform_distribution, 
                                                                                    repeat_simulation,
                                                                                    "min_ptm")

    np.savez("../../output/evaluated_"+modform_file_name+"_data.npz", mass_shift, chi_sqaure_score, mass_shift_deviation, 
             ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10)
    
    
    ### simulated complex PTM patterns
    modform_file_name = "complex"
    modform_distribution = pd.read_csv("ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")
    modform_distribution["rel. intensity"] = modform_distribution["intensity"]/ modform_distribution["intensity"].sum()
    
    ### evaluate algorithm on overlapping isotopic ditribution data, how much overlap can be tolerated?
    # min_ptm
    mass_shift, chi_sqaure_score, mass_shift_deviation, ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10 = test_overlapping_mass_detection(
                                                                                    data_simulation, 
                                                                                    vertical_error_par, 
                                                                                    horizontal_error_par, 
                                                                                    basal_noise_par, 
                                                                                    modform_distribution, 
                                                                                    repeat_simulation,
                                                                                    "min_ptm")

    np.savez("../../output/evaluated_"+modform_file_name+"_min_ptm_data.npz", mass_shift, chi_sqaure_score, mass_shift_deviation, 
             ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10)
    
    # min_both
    mass_shift, chi_sqaure_score, mass_shift_deviation, ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10 = test_overlapping_mass_detection(
                                                                                    data_simulation, 
                                                                                    vertical_error_par, 
                                                                                    horizontal_error_par, 
                                                                                    basal_noise_par, 
                                                                                    modform_distribution, 
                                                                                    repeat_simulation,
                                                                                    "min_both")

    np.savez("../../output/evaluated_"+modform_file_name+"min_both_data.npz", mass_shift, chi_sqaure_score, mass_shift_deviation, 
             ptm_patterns, ptm_patterns_top3, ptm_patterns_top5, ptm_patterns_top10)

    ### simulated phospho patterns
    modform_file_name = "phospho"
    modform_distribution = pd.read_csv("ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")
    modform_distribution["rel. intensity"] = modform_distribution["intensity"]/ modform_distribution["intensity"].sum()
    
    ### evaluate algorithm on phospho patterns, which objective function is best?
    performance_df = variable_error_noise_performance(data_simulation, mod, modform_distribution, repeat_simulation,
                                                      vertical_error_par, horizontal_error_par, basal_noise_par)
   
    performance_df.to_csv("../../output/performance_"+modform_file_name+".csv", sep=",", index=False) 



