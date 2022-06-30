# -*- coding: utf-8 -*-
"""
Created on Mar 30 2022

@author: Marjan Faizi
"""

import glob
import re
from itertools import product
import numpy as np
from scipy import optimize
import random
import pandas as pd

import utils 
from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
import config

#*************************************************************************************************************#
#*************************************************************************************************************#
# This method estimates the following error and noise levels from given experimental data:                    #
# - basal noise from a mass range wihtout any signal                                                          #
# - peak width error of the single peaks within the mass range of interest                                #
# - vertical error is estimated by the distance between the experimental peaks and the fitted gaussian curve  #
# - horizontal error is calculated as the distance between one peak to another                                #
#*************************************************************************************************************#
#*************************************************************************************************************#

file_names = [file for file in glob.glob(config.file_names)] 

sample_names = [cond+"_"+rep for cond, rep in product(config.conditions, config.replicates)]
mass_start_range = 43700
mass_end_range = 44600
noise_start_range = 47000
noise_end_range = 49000

basal_noise = []
width_sinlge_peak = []
vertical_error = []
horizontal_error = []
signal = []

for rep in config.replicates:
    
    file_names_same_replicate = [file for file in file_names if re.search(rep, file)]

    for cond in config.conditions:
    
        sample_name = [file for file in file_names_same_replicate if re.search(cond, file)]        
        if sample_name:
            sample_name = sample_name[0]

            data = MassSpecData(sample_name)
            data.set_search_window_mass_range(mass_start_range, mass_end_range)
            data_in_search_window = data.determine_search_window(data.raw_spectrum)
            data_in_search_window[:,1] = data_in_search_window[:,1] / data_in_search_window[:,1].max()
            all_peaks = data.picking_peaks()
            peaks_in_search_window = data.determine_search_window(all_peaks)
            peaks_in_search_window[:,1] = data.normalize_intensities(peaks_in_search_window[:,1])
            
            noise_level =  config.noise_level_fraction*peaks_in_search_window[:,1].std()
            peaks_in_search_window_above_noise = peaks_in_search_window[peaks_in_search_window[:,1]>noise_level]

            gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
            gaussian_model.determine_variable_window_sizes(config.unmodified_species_mass, config.window_size_lb, config.window_size_ub)
            gaussian_model.fit_gaussian_within_window(peaks_in_search_window_above_noise, config.allowed_overlap_fitting_window, config.pvalue_threshold, noise_level)      
            gaussian_model.refit_results(peaks_in_search_window_above_noise, noise_level, refit_mean=True)

            x_gauss_func = peaks_in_search_window[:,0]
            y_gauss_func = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results["amplitude"], gaussian_model.fitting_results["mean"], gaussian_model.stddev)
            vertical_error += (y_gauss_func - peaks_in_search_window[:,1]).tolist()

            spacing_between_peaks = 1.003355
            horizontal_error += ((peaks_in_search_window[1:,0]-peaks_in_search_window[:-1,0]) - spacing_between_peaks).tolist()

            for peak in peaks_in_search_window:
                window_size = 0.6
                selected_region_ix = np.argwhere((data_in_search_window[:,0]<=peak[0]+window_size) & (data_in_search_window[:,0]>=peak[1]-window_size))[:,0]
                selected_region = data_in_search_window[selected_region_ix]
                guess = 0.2
                optimized_param, _ = optimize.curve_fit(lambda x, sigma: utils.gaussian(x, peak[1], peak[0], sigma), 
                                                        selected_region[:,0], selected_region[:,1], maxfev=100000,
                                                        p0=guess, bounds=(0, window_size))
                width_sinlge_peak.append(optimized_param[0])

            noise_start_ix = data.find_nearest_mass_idx(all_peaks, noise_start_range)
            noise_end_ix = data.find_nearest_mass_idx(all_peaks, noise_end_range)
            basal_noise += random.sample((all_peaks[noise_start_ix:noise_end_ix,1]/data.rescaling_factor).tolist(), peaks_in_search_window.shape[0])

            signal += (peaks_in_search_window[:,1]>noise_level).tolist()

error_estimate_table = pd.DataFrame()

error_estimate_table["peak width"] = width_sinlge_peak
error_estimate_table["horizontal error (Da)"] = horizontal_error+len(sample_names)*[0]
error_estimate_table["vertical error (rel.)"] = vertical_error
error_estimate_table["basal noise (a.u.)"] = basal_noise
error_estimate_table["is_signal"] = signal

error_estimate_table.to_csv("../output/error_noise_distribution_table_06_29_22.csv", index=False)




