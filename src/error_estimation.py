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
# - width, called sigma, of the single peaks within the mass range of interest                                #
# - vertical error is estimated by the distance between the experimental peaks and the fitted gaussian curve  #
# - horizontal error is calculated as the distance between one peak to another                                #
#*************************************************************************************************************#
#*************************************************************************************************************#

file_names = [file for file in glob.glob(config.file_names)] 

sample_names = [cond+"_"+rep for cond, rep in product(config.conditions, config.replicates)]
mass_start_range = 43600
mass_end_range = 44600
noise_start_range = 47000
noise_end_range = 49000

basal_noise = []
sigma_sinlge_peak = []
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
            gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
            gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)
            gaussian_model.fit_gaussian_to_single_peaks(peaks_in_search_window, noise_level, config.pvalue_threshold)
            gaussian_model.remove_overlapping_fitting_results()
            gaussian_model.refit_results(peaks_in_search_window, noise_level, refit_mean=True)

            x_gauss_func = peaks_in_search_window[:,0]
            y_gauss_func = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results["amplitude"], gaussian_model.fitting_results["mean"], gaussian_model.stddev)
            vertical_error += (y_gauss_func / peaks_in_search_window[:,1]).tolist()

            proton_mass = 1.007
            horizontal_error += ((peaks_in_search_window[1:,0]-peaks_in_search_window[:-1,0]) - proton_mass).tolist()

            for peak in peaks_in_search_window:
                window_size = 0.6
                selected_region_ix = np.argwhere((data_in_search_window[:,0]<=peak[0]+window_size) & (data_in_search_window[:,0]>=peak[1]-window_size))[:,0]
                selected_region = data_in_search_window[selected_region_ix]
                guess = 0.2
                optimized_param, _ = optimize.curve_fit(lambda x, sigma: utils.gaussian(x, peak[1], peak[0], sigma), 
                                                        selected_region[:,0], selected_region[:,1], maxfev=100000,
                                                        p0=guess, bounds=(0, window_size))
                sigma_sinlge_peak.append(optimized_param[0])

            noise_start_ix = data.find_nearest_mass_idx(all_peaks, noise_start_range)
            noise_end_ix = data.find_nearest_mass_idx(all_peaks, noise_end_range)
            basal_noise += random.sample((all_peaks[noise_start_ix:noise_end_ix,1]/data.rescaling_factor).tolist(), peaks_in_search_window.shape[0])

            signal += (peaks_in_search_window[:,1]>noise_level).tolist()

error_estimate_table = pd.DataFrame()

error_estimate_table["sigma noise"] = sigma_sinlge_peak
error_estimate_table["horizontal noise (Da)"] = horizontal_error+len(sample_names)*[0]
error_estimate_table["vertical noise (rel.)"] = vertical_error
error_estimate_table["basal noise (a.u.)"] = basal_noise
error_estimate_table["is_signal"] = signal

error_estimate_table.to_csv("../output/noise_distribution_table.csv", index=False)


###########################################################################################################################################
error_estimate_table = pd.read_csv("../output/noise_distribution_table.csv")

import seaborn as sns
import matplotlib.pyplot as plt


fig, axes = plt.subplots(2, 2, figsize=(8,6))
sns.histplot(x="basal noise (a.u.)", data=error_estimate_table[error_estimate_table["basal noise (a.u.)"]<0.12], bins=50, ax=axes[0][0])
sns.histplot(x="sigma noise", data=error_estimate_table[error_estimate_table["sigma noise"]<0.55], bins=50, ax=axes[0][1])
#sns.histplot(x="sigma noise", data=error_estimate_table[(error_estimate_table["sigma noise"]<0.55) & (error_estimate_table["is_signal"]==True)], bins=50, ax=axes[0][1], color="orange")
sns.histplot(x="horizontal noise (Da)", data=error_estimate_table[error_estimate_table["horizontal noise (Da)"]<1], bins=50, ax=axes[1][0])
#sns.histplot(x="horizontal noise (Da)", data=error_estimate_table[(error_estimate_table["horizontal noise (Da)"]<1) & (error_estimate_table["is_signal"]==True)], bins=50, ax=axes[1][0], color="orange")
sns.histplot(x="vertical noise (rel.)", data=error_estimate_table[error_estimate_table["vertical noise (rel.)"]<2.5], bins=50, ax=axes[1][1])
#sns.histplot(x="vertical noise (rel.)", data=error_estimate_table[(error_estimate_table["vertical noise (rel.)"]<2.5) & (error_estimate_table["is_signal"]==True)], bins=50, ax=axes[1][1], color="orange")
sns.despine()
plt.tight_layout()
plt.savefig("../output/noise_distribution.pdf")






