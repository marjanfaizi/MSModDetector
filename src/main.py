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

sys.path.append("..")
import config



if __name__ == "__main__":
    
    print("\n"+"-"*63)     
    print("\nLoad files...")

    protein_entries = utils.read_fasta("../"+config.fasta_file_name)
    file_names = [file for file in glob.glob("../"+config.file_names)] 

    protein_sequence = list(protein_entries.values())[0]
    unmodified_species_mass, stddev_isotope_distribution = utils.isotope_distribution_fit_par(protein_sequence, 100)
    mass_tolerance = config.mass_error_ppm*1e-6*unmodified_species_mass
    
    if not file_names:
        print("\nFiles do not exist.\n")
        sys.exit()
        
    mod = Modifications("../"+config.modfication_file_name, protein_sequence)

    mass_shifts = MassShifts(config.mass_range_start, config.mass_range_end)
    
    print("\nDetecting mass shifts...")
    
    stdout_text = []

    progress_bar_count = 0
    
    sample_names= [cond+"_"+rep for cond, rep in product(config.conditions, config.replicates)]
    parameter = pd.DataFrame(index=["noise_level", "rescaling_factor", "total_protein_abundance"], columns=sample_names)
    
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

            data = MassSpecData()
            data.add_raw_spectrum(sample_name)
            data.set_mass_range_of_interest(config.mass_range_start, config.mass_range_end)  
            all_peaks = data.picking_peaks()
            peaks_normalized = data.preprocess_peaks(all_peaks, config.distance_threshold_adjacent_peaks)

            if peaks_normalized.size:
                noise_level = config.noise_level_fraction*peaks_normalized[:,1].std()

                parameter.loc["noise_level", cond+"_"+rep] = noise_level
                parameter.loc["rescaling_factor", cond+"_"+rep] = data.rescaling_factor

                if len(peaks_normalized[peaks_normalized[:,1]>noise_level]):  
                    # 1. ASSUMPTION: The isotopic distribution follows a normal distribution.
                    # 2. ASSUMPTION: The standard deviation does not change when modifications are included to the protein mass. 
                    gaussian_model = GaussianModel(cond, stddev_isotope_distribution, config.window_size)
                    gaussian_model.fit_gaussian_within_window(peaks_normalized, noise_level, config.pvalue_threshold, config.allowed_overlap)      

                    gaussian_model.refit_results(peaks_normalized, noise_level, refit_mean=True)
                    gaussian_model.calculate_relative_abundaces(data.search_window_start, data.search_window_end)
                    parameter.loc["total_protein_abundance", cond+"_"+rep] = gaussian_model.total_protein_abundance  
         
                    mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, cond+"_"+rep)
 
                else:
                    stdout_text.append("No peaks above the SN threshold could be detected within the search window for the following condition: " + cond + "_" + rep)
            else:
                stdout_text.append("No peaks could be detected within the search window for the following condition: " + cond + "_" + rep)

            progress_bar_count += 1        
            utils.progress(progress_bar_count, len(config.replicates)*len(config.conditions))

    seperator_stdout_text = "\n"
    print(seperator_stdout_text.join(stdout_text))

    if mass_shifts.identified_masses_df.empty:
        print("\nNo masses detected.")
    else:
        mass_shifts.calculate_avg_mass()
        if config.bin_peaks == True:
            bin_size = 2*mass_tolerance
            mass_shifts.bin_peaks(bin_size)

        if config.calculate_mass_shifts == True:
            mass_shifts.add_mass_shifts(unmodified_species_mass)

        if config.determine_ptm_patterns == True:
            print("\nSearching for PTM combinations:")

            mass_shifts.determine_ptm_patterns(mod, mass_tolerance, config.objective_fun, config.laps_run_lp)        
            mass_shifts.add_ptm_patterns_to_table()
            mass_shifts.ptm_patterns_df.to_csv( "../output/ptm_patterns_table.csv", sep=',', index=False)
          
        mass_shifts.save_tables("../output/")
        parameter.to_csv("../output/parameter.csv", sep=",") 

    print(63*"-"+"\n\n")

