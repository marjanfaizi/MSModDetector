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

# Should be changed for different configurations
import myconfig as config

file_names = [file for file in glob.glob(config.path+config.file_name_ending)] 

if __name__ == "__main__":
    
    print("\n"+"-"*80)     
    print("\nLoad files...")
    
    if not file_names:
        print("\nFiles do not exist.\n")
        sys.exit()
    
    mod = Modifications(config.modfication_file_name)

    mass_shifts = MassShifts()
    
    print("\nDetecting mass shifts...")
    
    stdout_text = []

    progress_bar_count = 0
    
    sample_names= [cond+"_"+rep for cond, rep in product(config.conditions, config.replicates)]
    parameter = pd.DataFrame(columns=sample_names)
    
    for rep in config.replicates:
        
        file_names_same_replicate = [file for file in file_names if re.search(rep, file)]
        
        for cond in config.conditions:
        
            sample_name = [file for file in file_names_same_replicate if re.search(cond, file)]    
           
            if not sample_name:
                print("\nCondition not available.\n")
                sys.exit()
            else:
                sample_name = sample_name[0]

            data = MassSpecData(sample_name)
            data.set_search_window_mass_range(config.start_mass_range, config.max_mass_shift)        

            max_intensity = data.intensities.max()
            data.convert_to_relative_intensities(max_intensity)
            rescaling_factor = max_intensity/100.0

            all_peaks = data.picking_peaks()
            trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
            trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)

            if trimmed_peaks_in_search_window.size:
                sn_threshold = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
                parameter.loc[0, cond+"_"+rep] = sn_threshold*rescaling_factor

                trimmed_peaks_above_sn_in_search_window = trimmed_peaks_in_search_window[trimmed_peaks_in_search_window[:,1]>sn_threshold]

                if trimmed_peaks_above_sn_in_search_window.size:  
                    # 1. ASSUMPTION: The isotopic distribution follows a normal distribution.
                    # 2. ASSUMPTION: The standard deviation does not change when modifications are included to the protein mass. 
                    gaussian_model = GaussianModel(cond)
                    gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass_init)
                    gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, trimmed_peaks_above_sn_in_search_window)

                    gaussian_model.filter_fitting_results(config.pvalue_threshold)
                    gaussian_model.refit_amplitudes(trimmed_peaks_in_search_window, sn_threshold)
                    gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)
         
                    mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, rescaling_factor, cond+"_"+rep)
 
                else:
                    stdout_text.append("No peaks above the SN threshold could be detected within the search window for the following condition: " + cond + "_" + rep)
            else:
                stdout_text.append("No peaks could be detected within the search window for the following condition: " + cond + "_" + rep)

            progress_bar_count += 1        
            utils.progress(progress_bar_count, len(file_names))

    print("\n")
    seperator_stdout_text = "\n"
    print(seperator_stdout_text.join(stdout_text))

    if mass_shifts.identified_masses_df.empty:
        print("\nNo masses detected.")
    else:
        calibration_number_dict = mass_shifts.align_spetra()
        parameter = parameter.append(calibration_number_dict, ignore_index=True)
        mass_shifts.bin_peaks()

        if config.calculate_mass_shifts == True:
            mass_shifts.add_mass_shifts()

        if config.determine_ptm_patterns == True:
            print("\nSearching for PTM combinations:")

            mass_shifts.determine_ptm_patterns(mod, config.mass_tolerance)        
            mass_shifts.add_ptm_patterns_to_table()
            mass_shifts.save_table(mass_shifts.ptm_patterns_df, "../output/ptm_patterns_table.csv")
            
        mass_shifts.save_table(mass_shifts.identified_masses_df, "../output/mass_shifts.csv")
        parameter.rename(index={0:"noise_level", 1:"calibration_number"}, inplace=True)
        parameter.to_csv("../output/parameter.csv", sep=",") 

    print("\n")
    print(80*"-"+"\n\n")



