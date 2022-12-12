#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2021

@author: Marjan Faizi
"""

import glob
import sys
import argparse
import re
from itertools import product
import pandas as pd

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
import utils

#sys.path.append("..")
#sys.path.append("simulated_data/")
#import config


######
parser = argparse.ArgumentParser(prog='PROG',
                                 description="MSModDetector detects and quantifies mass shifts for a protein of interest and subsequently infers potential PTM patterns using linear programming.")


"""
# This regular expression specifies the ending of the file names that should be read all at once
file_names = "raw_data/*.mzml"

# list of all replicate names as they are in the file names
replicates = ["rep5"]#, "rep6"] # rep1, rep5, rep6, rep9

# list of all condition names as they are in the file names
#conditions = ["nutlin_only", "xray_2hr", "xray_7hr", "xray-nutlin", "uv_7hr"]
conditions = ["nutlin_only", "xray_2hr", "xray_7hr"]





"""
parser.add_argument("-mod", type=str, help="Path to the table with all modification types.")
parser.add_argument("-fasta", type=str, help="Path to fasta file for protein of interest.")
parser.add_argument("-start", type=float, help="Set start of mass range to search for shifts.")
parser.add_argument("-end", type=float, help="Set end of mass range to search for shifts.")
parser.add_argument("-wsize", type=float,
                    help="Set window size used to fit the gaussian distribution.")
parser.add_argument("-ol",  default=0.3, type=float,
                    help="Set how much overlap should be allowed between two isotopic distributions in percentage.")
parser.add_argument("-nfrac",  default=0.5, type=float,
                    help="The standard deviation of the data points within the search window determines the noise level. The threshold of the noise level can be decreased with this parameter.")
parser.add_argument("-err",  default=36.0, type=float,
                    help="Mass error in ppm.")
parser.add_argument("-bin",  default=True, type=bool,
                    help="Mass shifts across samples will be binned together.")
parser.add_argument("-laps",  default=5, type=int,
                    help="Solve optimization k times and report the best k optimal solutions.")
parser.add_argument("-obj", default="min_ptm", 
                    help="Choose between three objective functions. 1) min_ptm: minimize total amount of PTMs on a single protein, 2) min_err: minimize error between observed and inferred mass shift, 3) min_both: minimize error and total amount of PTMs.")
parser.add_argument("-pval",  default=0.99999, type=float,
                    help="The fit of the gaussian distribution to the observed isotope distribution is evaluated by the chi-squared test. A high p-value indicates a better fit; both distributions are less likely to differ from each other.")
parser.add_argument("-ms", default=True, type=bool,
                    help="Set this to True if the mass shifts should be calculated and reported in the output table.")
parser.add_argument("-ptm", default=True, type=bool,
                    help="Set this to True if the PTM patterns should be determined and reported in the output table.")

args = parser.parse_args()
#########

"""
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
            peaks_normalized = data.preprocess_peaks(all_peaks)

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

"""