#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 1 2022

@author: Marjan Faizi
"""

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
import config
import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob


plt.style.use("seaborn-pastel")
sns.set_color_codes("colorblind")
sns.set_context("paper", font_scale=1.2)


################################################################################################
####################################### RAW SPECTRUM ###########################################
################################################################################################
sample_name = "xray_7hr_rep5"
file_name = "../data/raw_data/P04637/" + sample_name + ".mzml"
data = MassSpecData(file_name)

plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], color="0.3", linewidth=1)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((2000,55000))
plt.ylim(ymin=-200)
plt.tight_layout()
plt.savefig('../output/raw_spectrum.pdf', dpi=800)
plt.show()

plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], color="0.3", linewidth=1)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43550,44900))
plt.ylim((-20,950))
plt.tight_layout()
plt.savefig('../output/raw_spectrum_zoomed.pdf', dpi=800)
plt.show()
################################################################################################
################################################################################################



################################################################################################
#################################### DATA PREPROCESSING ########################################
################################################################################################
distribution_start = 43755
distribution_end = 43812

plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], ".-", color="0.3",
         markersize=1.8, linewidth=0.8)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((distribution_start,distribution_end))
plt.ylim((-20,780))
plt.tight_layout()
plt.savefig('../output/isotopic_distribution.pdf', dpi=800)
plt.show()

all_peaks = data.picking_peaks()
trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)  

plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], ".-", color="0.3",
         markersize=1.8, linewidth=0.8)
plt.plot(trimmed_peaks[:,0], trimmed_peaks[:,1], ".", color="r", markersize=2.5)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((distribution_start,distribution_end))
plt.ylim((-20,780))
plt.tight_layout()
plt.savefig('../output/isotopic_distribution_peaks_only.pdf', dpi=800)
plt.show()

data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)      
trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])
noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
trimmed_peaks_above_noise_in_search_window = trimmed_peaks_in_search_window[trimmed_peaks_in_search_window[:,1]>noise_level]

plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], ".-", color="0.3",
         markersize=1.8, linewidth=0.8)
plt.plot(trimmed_peaks_in_search_window[:,0], data.rescaling_factor*trimmed_peaks_in_search_window[:,1], 
         ".", color="b", markersize=2.5)
plt.plot(trimmed_peaks_above_noise_in_search_window[:,0], 
         data.rescaling_factor*trimmed_peaks_above_noise_in_search_window[:,1], 
         ".", color="r", markersize=2.5)
plt.axhline(y=noise_level*data.rescaling_factor, color='b', linestyle='-')
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((distribution_start,distribution_end))
plt.ylim((-20,780))
plt.tight_layout()
plt.savefig('../output/isotopic_distribution_normalized_above_sn_threshold.pdf', dpi=800)
plt.show()
################################################################################################
################################################################################################



################################################################################################
################################# FITTING OF GAUSSIAN MODEL ####################################
################################################################################################
gaussian_model = GaussianModel(sample_name, config.stddev_isotope_distribution)
gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)

for peak in trimmed_peaks_above_noise_in_search_window:     
    mass = peak[0]
    if mass > distribution_start and mass < distribution_end: 
        
        for window_size in gaussian_model.fitting_window_sizes:
            selected_region_ix = np.argwhere((trimmed_peaks_in_search_window[:,0]<=mass+window_size) & 
                                             (trimmed_peaks_in_search_window[:,0]>=mass-window_size))[:,0]
            selected_region = trimmed_peaks_in_search_window[selected_region_ix]
            sample_size = selected_region.shape[0]          
            
            if sample_size >= gaussian_model.sample_size_threshold:
                fitted_amplitude = gaussian_model.fit_gaussian(selected_region, mean=mass)[0]
                chi_square_score = gaussian_model.chi_square_test(selected_region, fitted_amplitude, mass)          
                
            predicted_intensities = utils.gaussian(selected_region[:,0], fitted_amplitude, mass, gaussian_model.stddev)  
            plt.figure(figsize=(7,3))
            plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1]/data.rescaling_factor, ".-", color="0.3",
                     markersize=1.8, linewidth=0.8)
            plt.plot(selected_region[:,0], predicted_intensities, "-", color="r", linewidth=0.8)
            plt.plot(trimmed_peaks_in_search_window[:,0], trimmed_peaks_in_search_window[:,1], 
                     ".", color="b", markersize=2.5)
            plt.plot(trimmed_peaks_above_noise_in_search_window[:,0], 
                     trimmed_peaks_above_noise_in_search_window[:,1], 
                     ".", color="r", markersize=2.5)
            plt.axvline(mass-window_size, color='r', linestyle='-')
            plt.axvline(mass+window_size, color='r', linestyle='-')
            plt.axhline(y=noise_level, color='b', linestyle='-')
            plt.text(mass, fitted_amplitude*1.1, str(round(chi_square_score.statistic,2))+', '+str(round(chi_square_score.pvalue,2)), fontsize=7)
            plt.xlabel("mass (Da)")
            plt.ylabel("intensity (a.u.)")
            plt.xlim((distribution_start-5,distribution_end+5))
            plt.ylim((-0.025,1.1))
            plt.tight_layout()
            plt.savefig("../output/video/gaussian_model"+str(mass)+str(int(window_size))+".jpg", dpi=800)


img_array = []
for filename in np.sort(glob.glob("../output/video/*.jpg")):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

video = cv2.VideoWriter("../output/video/fitting.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 2.5, size)

for i in range(len(img_array)):
    video.write(img_array[i])
video.release()


fitting_results = pd.DataFrame(columns=["mean", "amplitude", "chi_score", "p_value", "window_size"]) 

for peak in trimmed_peaks_above_noise_in_search_window:     
    mass = peak[0]
    
    for window_size in gaussian_model.fitting_window_sizes:
        selected_region_ix = np.argwhere((trimmed_peaks_in_search_window[:,0]<=mass+window_size) & 
                                         (trimmed_peaks_in_search_window[:,0]>=mass-window_size))[:,0]
        selected_region = trimmed_peaks_in_search_window[selected_region_ix]
        sample_size = selected_region.shape[0]              
        
        if sample_size >= gaussian_model.sample_size_threshold:
            fitted_amplitude = gaussian_model.fit_gaussian(selected_region, mean=mass)[0]
            chi_square_score = gaussian_model.chi_square_test(selected_region, fitted_amplitude, mass)
            pvalue = chi_square_score.pvalue
            chi_score = chi_square_score.statistic
            fitting_results = fitting_results.append({"mean": mass, "amplitude": fitted_amplitude, "chi_score": chi_score,
                                                      "p_value": pvalue, "window_size": window_size}, ignore_index=True)

#sns.boxplot(x=2*fitting_results["window_size"], y=np.log2(fitting_results["chi_score"]))
#sns.boxplot(x=2*fitting_results["window_size"], y=fitting_results["p_value"])

gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, config.pvalue_threshold)

plt.figure(figsize=(4,3))
gaussian_model.fitting_results["window_size"].hist(grid=False, color="0.5")
plt.xlabel("window size")
plt.tight_layout()
plt.savefig('../output/window_sizes.pdf', dpi=800)
plt.show()


gaussian_model.remove_overlapping_fitting_results()
gaussian_model.refit_amplitudes(trimmed_peaks_in_search_window, noise_level)
gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)

        
x_gauss_func = np.arange(config.mass_start_range, config.mass_end_range)
y_gauss_func = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results["amplitude"].values, 
                                    gaussian_model.fitting_results["mean"].values, config.stddev_isotope_distribution)
plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1]/data.rescaling_factor, ".-", color="0.3",
         markersize=1.8, linewidth=0.8)
plt.plot(gaussian_model.fitting_results["mean"].values, gaussian_model.fitting_results["amplitude"].values, ".", color="r")
plt.plot(x_gauss_func, y_gauss_func, color='r')
plt.axhline(y=noise_level, color='b', linestyle='-')
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((distribution_start-5,distribution_end+5))
plt.ylim((-0.025,1.1))
plt.tight_layout()
plt.savefig("../output/best_fit.pdf", dpi=800)
################################################################################################
################################################################################################



################################################################################################
################################ COMPARISON OF MULTIPLE SAMPLES ################################
################################################################################################

# once the mass shifts for the reference protein mass are detected go to next topic: comparison of mutliple samples
# show how the bin size effects the results of the mass shifts

################################################################################################
################################################################################################



################################################################################################
############################## MILP AND INFERENCE OF PTM PATTERNS ##############################
################################################################################################

# next step: infer ptm patterns 
# explain how we use MILP
# use different objective functions
# use the MAPK1 example as toy example

# show how the number of combinations changes for differentmass error tolerances for p53
# output in the end: list of ptm pattern combinatons for each mass shift ranked according to the objective function 


################################################################################################
################################################################################################


