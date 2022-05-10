#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 5 2022

@author: Marjan Faizi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
import config
import utils

sns.set()
sns.set_style("white")
################################################################################################
############################################# FIGURE 1 #########################################
################################################################################################

file_name = "../data/raw_data/P04637/xray_7hr_rep5.mzml"
rep = "rep5"
cond = "xray_7hr"

data = MassSpecData(file_name)
data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)        


plt.figure(figsize=(2.,1.6))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '-', color="0.3", linewidth=1)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("intensity (a.u.)", fontsize=10)
plt.xlim((43755,43800))
plt.ylim((-10, 800))
plt.tight_layout()
sns.despine()
plt.show()


all_peaks = data.picking_peaks()
trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])
noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()

peaks_above_noise_level = trimmed_peaks_in_search_window[:,1]>noise_level

plt.figure(figsize=(2.,1.6))
plt.plot(trimmed_peaks_in_search_window[:,0], trimmed_peaks_in_search_window[:,1], '.', color="0.6")
plt.plot(trimmed_peaks_in_search_window[peaks_above_noise_level,0], trimmed_peaks_in_search_window[peaks_above_noise_level,1], '.', color="0.3")
plt.axhline(y=noise_level, c="b", lw=0.8)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("rel. intensity", fontsize=10)
plt.xlim((43755,43800))
plt.tight_layout()
sns.despine()
plt.show()


gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)
gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, config.pvalue_threshold)      
gaussian_model.remove_overlapping_fitting_results()

mean = gaussian_model.fitting_results["mean"]
amplitude = gaussian_model.fitting_results["amplitude"]
x = np.arange(43755,43800)

plt.figure(figsize=(2.,1.6))
plt.plot(trimmed_peaks_in_search_window[:,0], trimmed_peaks_in_search_window[:,1], '.', color="0.6")
plt.plot(trimmed_peaks_in_search_window[peaks_above_noise_level,0], trimmed_peaks_in_search_window[peaks_above_noise_level,1], '.', color="0.3")
plt.plot(x, utils.gaussian(x, amplitude[0], mean[0], gaussian_model.stddev), "r-", lw=0.95)
plt.plot(x, utils.gaussian(x, amplitude[1], mean[1], gaussian_model.stddev), "r-", lw=0.95)
plt.axhline(y=noise_level, c='b', lw=0.5)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("rel. intensity", fontsize=10)
plt.xlim((43755,43800))
plt.tight_layout()
sns.despine()
plt.show()


plt.figure(figsize=(2.,1.6))
plt.plot(trimmed_peaks_in_search_window[:,0], trimmed_peaks_in_search_window[:,1], '.', color="0.6")
plt.plot(trimmed_peaks_in_search_window[peaks_above_noise_level,0], trimmed_peaks_in_search_window[peaks_above_noise_level,1], '.', color="0.3")
plt.plot(x, utils.gaussian(x, amplitude[0], mean[0], gaussian_model.stddev), "r-", lw=0.95)
plt.plot(x, utils.gaussian(x, amplitude[1], mean[1], gaussian_model.stddev), "r-", lw=0.95)
plt.axvline(x=mean[0], c="k", lw=0.5, ls="--")
plt.axvline(x=mean[1], c="k", lw=0.5, ls="--")
plt.plot(mean[0], amplitude[0], "r.")
plt.plot(mean[1], amplitude[1], "r.")
plt.axhline(y=noise_level, c='b', lw=0.5)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("rel. intensity", fontsize=10)
plt.xlim((43755,43800))
plt.tight_layout()
sns.despine()
plt.show()


gaussian_model.refit_results(trimmed_peaks_in_search_window, noise_level, refit_mean=True)
gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)

plt.figure(figsize=(2.,1.6))
plt.plot(trimmed_peaks_in_search_window[:,0], trimmed_peaks_in_search_window[:,1], '.', color="0.6")
plt.plot(trimmed_peaks_in_search_window[peaks_above_noise_level,0], trimmed_peaks_in_search_window[peaks_above_noise_level,1], '.', color="0.3")
plt.plot(x, utils.gaussian(x, amplitude[0], mean[0], gaussian_model.stddev), "orange", lw=0.9)
plt.plot(x, utils.gaussian(x, amplitude[1], mean[1], gaussian_model.stddev), "plum", lw=0.9)
plt.fill_between(x,utils.gaussian(x, amplitude[0], mean[0], gaussian_model.stddev), color="none", hatch="//////", edgecolor="orange")
plt.fill_between(x,utils.gaussian(x, amplitude[1], mean[1], gaussian_model.stddev), color="none", hatch="//////", edgecolor="plum")
plt.plot(x, utils.multi_gaussian(x, amplitude, mean, gaussian_model.stddev), "r-")
plt.axhline(y=noise_level, c='b', lw=0.5)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("rel. intensity", fontsize=10)
plt.xlim((43755,43800))
plt.tight_layout()
sns.despine()
plt.show()  

################################################################################################
################################################################################################