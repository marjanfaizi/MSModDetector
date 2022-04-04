#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2021

@author: Marjan Faizi
"""

import glob
import sys
import re
from itertools import product
import pandas as pd

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
import utils
#import config_sim as config
import config

file_names = [file for file in glob.glob(config.file_names)] 

if __name__ == "__main__":
    
    print("\n"+"-"*63)     
    print("\nLoad files...")
    
    if not file_names:
        print("\nFiles do not exist.\n")
        sys.exit()
    
    aa_sequence_str = utils.read_fasta(config.fasta_file_name)
    mod = Modifications(config.modfication_file_name, aa_sequence_str)

    mass_shifts = MassShifts()
    
    print("\nDetecting mass shifts...")
    
    stdout_text = []

    progress_bar_count = 0
    
    sample_names= [cond+"_"+rep for cond, rep in product(config.conditions, config.replicates)]
    parameter = pd.DataFrame(columns=sample_names)
    
    for rep in config.replicates:
        
        file_names_same_replicate = [file for file in file_names if re.search(rep, file)]
        
        if not file_names_same_replicate:
                print("\nReplicate or condition not available. Adjust config.py file.\n")
                sys.exit()
        for cond in config.conditions:
        
            sample_name = [file for file in file_names_same_replicate if re.search(cond, file)]    
           
            if not sample_name:
                print("\nCondition not available. Adjust config.py file.\n")
                sys.exit()
            else:
                sample_name = sample_name[0]

            data = MassSpecData(sample_name)
            data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)        

            all_peaks = data.picking_peaks()
            trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
            trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
            trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])


            if trimmed_peaks_in_search_window.size:
                noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
                parameter.loc[0, cond+"_"+rep] = noise_level*data.rescaling_factor

                if any(trimmed_peaks_in_search_window[:,1]>noise_level):  
                    # 1. ASSUMPTION: The isotopic distribution follows a normal distribution.
                    # 2. ASSUMPTION: The standard deviation does not change when modifications are included to the protein mass. 
                    gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
                    gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)
                    gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, config.pvalue_threshold)      
                    gaussian_model.remove_overlapping_fitting_results()
                    gaussian_model.refit_results(trimmed_peaks_in_search_window, noise_level, refit_mean=True)
                    gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)
         
                    mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, data.rescaling_factor, cond+"_"+rep)
 
                else:
                    stdout_text.append("No peaks above the SN threshold could be detected within the search window for the following condition: " + cond + "_" + rep)
            else:
                stdout_text.append("No peaks could be detected within the search window for the following condition: " + cond + "_" + rep)

            progress_bar_count += 1        
            utils.progress(progress_bar_count, len(file_names))

    seperator_stdout_text = "\n"
    print(seperator_stdout_text.join(stdout_text))

    if mass_shifts.identified_masses_df.empty:
        print("\nNo masses detected.")
    else:
#        calibration_number_dict = mass_shifts.align_spetra()
#        parameter = parameter.append(calibration_number_dict, ignore_index=True)
#        parameter.rename(index={0:"noise_level", 1:"calibration_number"}, inplace=True)
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
        
        parameter.rename(index={0:"noise_level"}, inplace=True)
        parameter.to_csv("../output/parameter.csv", sep=",") 

    print(63*"-"+"\n\n")

