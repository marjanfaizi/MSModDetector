# -*- coding: utf-8 -*-
"""
Created on Mar 30 2022

@author: Marjan Faizi
"""

import glob
import re
from itertools import product
import numpy as np
import random
import pandas as pd

from src.mass_spec_data import MassSpecData
from src.gaussian_model import GaussianModel
from src import utils 

#*************************************************************************************************************#
#*************************************************************************************************************#
# This method estimates the following error and noise levels from given experimental data:                    #
# - basal noise from a mass range wihtout any signal                                                          #
# - vertical error is estimated by the distance between the experimental peaks and the fitted gaussian curve  #
# - horizontal error is calculated as the distance between one peak to another                                #
#*************************************************************************************************************#
#*************************************************************************************************************#

file_names = "../../raw_data/*.mzml"
replicates = ["rep5", "rep6"]
conditions = ["nutlin_only", "uv_7hr"]
distance_threshold_adjacent_peaks = 0.6
noise_level_fraction = 0.5
pvalue_threshold = 0.1
window_size_lb = 0.2
window_size_ub = 0.33
unmodified_species_mass = 43652.55 # mass of p53
stddev_isotope_distribution = 5.59568 # standard deviation for the isotopic distributio of unmodified p53


file_names = [file for file in glob.glob(file_names)] 
sample_names = [cond+"_"+rep for cond, rep in product(conditions, replicates)]

mass_range_start = 43700
mass_range_end = 44600
noise_range_start = 44600
noise_range_end = 46000

basal_noise = []
vertical_error = []
horizontal_error = []
signal = []

for rep in replicates:
    
    file_names_same_replicate = [file for file in file_names if re.search(rep, file)]

    for cond in conditions:
    
        sample_name = [file for file in file_names_same_replicate if re.search(cond, file)]        
        if sample_name:
            sample_name = sample_name[0]
            data = MassSpecData()
            data.add_raw_spectrum(sample_name)
            data.set_mass_range_of_interest(mass_range_start, mass_range_end) 
            all_peaks = data.picking_peaks()
            peaks_normalized = data.preprocess_peaks(all_peaks, distance_threshold_adjacent_peaks)
            noise_level = noise_level_fraction*peaks_normalized[:,1].std()
            peaks_normalized_above_noise = peaks_normalized[peaks_normalized[:,1]>noise_level]
            
            gaussian_model = GaussianModel(cond, stddev_isotope_distribution)
            gaussian_model.determine_variable_window_sizes(unmodified_species_mass, window_size_lb, window_size_ub)
            gaussian_model.fit_gaussian_within_window(peaks_normalized_above_noise, pvalue_threshold)      
            gaussian_model.refit_results(peaks_normalized, noise_level, refit_mean=True)
           
            x_gauss_func = peaks_normalized[:,0]
            y_gauss_func = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results["amplitude"], gaussian_model.fitting_results["mean"], gaussian_model.stddev)
            vertical_error += (y_gauss_func - peaks_normalized[:,1]).tolist()

            spacing_between_peaks = 1.0
            horizontal_error += ((peaks_normalized[1:,0]-peaks_normalized[:-1,0]) - spacing_between_peaks).tolist()
            
            noise_start_ix = data.find_nearest_mass_idx(all_peaks, noise_range_start)
            noise_end_ix = data.find_nearest_mass_idx(all_peaks, noise_range_end)
            basal_noise += random.sample((all_peaks[noise_start_ix:noise_end_ix,1]/data.rescaling_factor).tolist(), peaks_normalized.shape[0])

            signal += (peaks_normalized[:,1]>noise_level).tolist()


error_estimate_table = pd.DataFrame()
error_estimate_table["horizontal_error"] =  np.pad(horizontal_error, (0, len(vertical_error)-len(horizontal_error)))
error_estimate_table["vertical_error"] = vertical_error
error_estimate_table["basal_noise"] = basal_noise
error_estimate_table["is_signal"] = signal
error_estimate_table.to_csv("../../output/error_noise_distribution_table.csv", index=False)




