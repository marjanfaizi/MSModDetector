#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 5 2022

@author: Marjan Faizi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyopenms import AASequence

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
import config
import utils

sns.set()
sns.set_style("white")


################################################################################################
############################################ TOY EXAMPLE #######################################
################################################################################################
file_name = "../data/raw_data/P04637/xray_7hr_rep5.mzml"
rep = "rep5"
cond = "xray_7hr"

data = MassSpecData(file_name)
data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)  
################################################################################################
################################################################################################


################################################################################################
############################################# FIGURE 1 #########################################
################################################################################################
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



################################################################################################
#################################### SUPPLEMENTAL FIGURE 1 #####################################
################################################################################################
plt.figure(figsize=(7, 3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '-', color="0.3", linewidth=0.8)
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("intensity (a.u.)", fontsize=10)
plt.xlim((2500, 80000))
plt.ylim((-300, 11500))
plt.tight_layout()
sns.despine()
plt.show()

plt.figure(figsize=(4.5, 1.8))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '-', color="0.3", linewidth=0.8)
plt.xlabel("mass (Da)", fontsize=9)
plt.ylabel("intensity (a.u.)", fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlim((43700, 44600))
plt.ylim((-20, 850))
plt.tight_layout()
sns.despine()
plt.show()

plt.figure(figsize=(4.5, 1.8))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '-', color="0.3", linewidth=0.8)
plt.xlabel("mass (Da)", fontsize=9)
plt.ylabel("intensity (a.u.)", fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlim((47000, 49000))
plt.ylim((-1, 52))
plt.tight_layout()
sns.despine()
plt.show()

################################################################################################
################################################################################################


################################################################################################
#################################### SUPPLEMENTAL FIGURE 2 #####################################
################################################################################################
protein_sequence = AASequence.fromString(utils.read_fasta(config.fasta_file_name))
distribution = utils.get_theoretical_isotope_distribution(protein_sequence, 100)

mass_grid = np.arange(distribution[0,0], distribution[-1,0], 0.02)
intensities = np.zeros(mass_grid.shape)
peak_width = 0.2
for peak in distribution:
    intensities += peak[1] * np.exp(-0.5*((mass_grid - peak[0]) / peak_width)**2)

mean_p53, stddev_p53 = utils.mean_and_stddev_of_isotope_distribution(config.fasta_file_name, 100)

plt.figure(figsize=(6, 3))
plt.plot(mass_grid, intensities, '-', color="0.3", label="isotopic distribution of p53")
plt.plot(mass_grid, utils.gaussian(mass_grid, intensities.max(), mean_p53, stddev_p53), 'r-', 
         label="Gaussian fit") 
plt.plot(mean_p53, intensities.max(), "r.")
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("intensity (a.u.)", fontsize=10)
plt.xlim((43630, 43680))
plt.legend()
plt.tight_layout()
sns.despine()
plt.show()
################################################################################################
################################################################################################


################################################################################################
#################################### SUPPLEMENTAL FIGURE 3 #####################################
################################################################################################
sns.set_style("ticks")

error_estimate_table = pd.read_csv("../output/error_noise_distribution_table.csv")


def exponential(x, beta):
    return (1/beta)*np.exp(-(x/beta))


peak_width_mean = error_estimate_table[error_estimate_table["peak width"]<0.4]["peak width"].mean()
peak_width_std = error_estimate_table[error_estimate_table["peak width"]<0.4]["peak width"].std()


fig, axes = plt.subplots(2, 2, figsize=(7,5))
sns.histplot(x="basal noise (a.u.)", data=error_estimate_table, bins=100, ax=axes[0][0])
axes[0][0].plot(error_estimate_table["basal noise (a.u.)"].sort_values(), 20*exponential(error_estimate_table["basal noise (a.u.)"].sort_values(), 1/100), color="purple")
axes[0][0].plot(error_estimate_table["basal noise (a.u.)"].sort_values(), 10*exponential(error_estimate_table["basal noise (a.u.)"].sort_values(), 1/200), color="red")
axes[0][0].plot(error_estimate_table["basal noise (a.u.)"].sort_values(), 5*exponential(error_estimate_table["basal noise (a.u.)"].sort_values(), 1/400), color="brown")
axes[0][0].set_xlim([0,0.12])
#sns.histplot(x="basal noise (a.u.)", data=error_estimate_table[error_estimate_table["basal noise (a.u.)"]<0.12], bins=30, ax=axes[0][0])
sns.histplot(x="peak width", data=error_estimate_table[error_estimate_table["peak width"]<0.5], bins=50, ax=axes[0][1])
axes[0][1].plot(error_estimate_table["peak width"].sort_values(), utils.gaussian(error_estimate_table["peak width"].sort_values(), 300, peak_width_mean, peak_width_std), color="purple")
axes[0][1].plot(error_estimate_table["peak width"].sort_values(), utils.gaussian(error_estimate_table["peak width"].sort_values(), 300, peak_width_mean, peak_width_std/2), color="red")
axes[0][1].plot(error_estimate_table["peak width"].sort_values(), utils.gaussian(error_estimate_table["peak width"].sort_values(), 300, peak_width_mean, peak_width_std/4), color="brown")
axes[0][1].set_xlim([0.1,0.5])
sns.histplot(x="peak width", data=error_estimate_table[(error_estimate_table["peak width error"]<0.55) & (error_estimate_table["is_signal"]==True)], bins=50, ax=axes[0][1], color="orange")
sns.histplot(x="horizontal error (Da)", data=error_estimate_table[error_estimate_table["horizontal error (Da)"]<1], bins=50, ax=axes[1][0])
axes[1][0].plot(error_estimate_table["horizontal error (Da)"].sort_values(), utils.gaussian(error_estimate_table["horizontal error (Da)"].sort_values(), 1e3, 0, 0.1), color="purple")
axes[1][0].plot(error_estimate_table["horizontal error (Da)"].sort_values(), utils.gaussian(error_estimate_table["horizontal error (Da)"].sort_values(), 1e3, 0, 0.05), color="red")
axes[1][0].plot(error_estimate_table["horizontal error (Da)"].sort_values(), utils.gaussian(error_estimate_table["horizontal error (Da)"].sort_values(), 1e3, 0, 0.025), color="brown")
axes[1][0].set_xlim([-1,1])
sns.histplot(x="horizontal error (Da)", data=error_estimate_table[(error_estimate_table["horizontal error (Da)"]<1) & (error_estimate_table["is_signal"]==True)], bins=50, ax=axes[1][0], color="orange")
sns.histplot(x="vertical error (rel.)", data=error_estimate_table[error_estimate_table["vertical error (rel.)"]<2.5], bins=50, ax=axes[1][1])
plt.plot(error_estimate_table["vertical error (rel.)"].sort_values(), utils.gaussian(error_estimate_table["vertical error (rel.)"].sort_values(), 300, 1, 0.25), color="purple")
plt.plot(error_estimate_table["vertical error (rel.)"].sort_values(), utils.gaussian(error_estimate_table["vertical error (rel.)"].sort_values(), 300, 1, 0.125), color="red")
plt.plot(error_estimate_table["vertical error (rel.)"].sort_values(), utils.gaussian(error_estimate_table["vertical error (rel.)"].sort_values(), 300, 1, 0.0625), color="brown")
axes[1][1].set_xlim([0,2.5])
axes[1][1].set_ylim([-1,600])
sns.histplot(x="vertical error (rel.)", data=error_estimate_table[(error_estimate_table["vertical error (rel.)"]<2.5) & (error_estimate_table["is_signal"]==True)], bins=50, ax=axes[1][1], color="orange")
sns.despine()
plt.tight_layout()
plt.show()

################################################################################################
################################################################################################


################################################################################################
#################################### SUPPLEMENTAL FIGURE 4 #####################################
################################################################################################


# spectrum mit different noise levels of phospho example

################################################################################################
################################################################################################


################################################################################################
#################################### SUPPLEMENTAL FIGURE 5 #####################################
################################################################################################


# spectrum of complex example with noise

################################################################################################
################################################################################################





