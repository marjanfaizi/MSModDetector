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
import itertools

from simulate_data import SimulateData
from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
import config
import utils

sns.set()
sns.set_style("white")


###################################################################################################################
############################################## INPUT AND TOY EXAMPLE ##############################################
###################################################################################################################
file_name = "../data/raw_data/P04637/xray_7hr_rep5.mzml"
rep = "rep5"
cond = "xray_7hr"

data = MassSpecData(file_name)
data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)  

aa_sequence_str = utils.read_fasta(config.fasta_file_name)
modifications_table = pd.read_csv(config.modfication_file_name, sep=';')
###################################################################################################################
###################################################################################################################


###################################################################################################################
#################################################### FIGURE 1 #####################################################
###################################################################################################################
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

###################################################################################################################
###################################################################################################################



###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 1 ##############################################
###################################################################################################################
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

###################################################################################################################
###################################################################################################################


###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 2 ##############################################
###################################################################################################################
sns.set_style("ticks")

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
plt.legend(frameon=False, fontsize=10)
plt.tight_layout()
sns.despine()
plt.show()
###################################################################################################################
###################################################################################################################


###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 3 ##############################################
###################################################################################################################
error_estimate_table = pd.read_csv("../output/error_noise_distribution_table.csv")


def exponential(x, beta):
    return (1/beta)*np.exp(-(x/beta))


peak_width_mean = error_estimate_table[(error_estimate_table["peak width"]<0.4) & (error_estimate_table["is_signal"]==True)]["peak width"].mean()
peak_width_std = error_estimate_table[(error_estimate_table["peak width"]<0.4) & (error_estimate_table["is_signal"]==True)]["peak width"].std()

lw = 1.3

fig, axes = plt.subplots(2, 2, figsize=(7,5))
sns.histplot(x="basal noise (a.u.)", data=error_estimate_table, bins=70, ax=axes[0][0])
#axes[0][0].plot(error_estimate_table["basal noise (a.u.)"].sort_values(), 8*exponential(error_estimate_table["basal noise (a.u.)"].sort_values(), 1/100), color="purple", lw=lw)
axes[0][0].plot(error_estimate_table["basal noise (a.u.)"].sort_values(), 5*exponential(error_estimate_table["basal noise (a.u.)"].sort_values(), 1/120), color="purple", lw=lw)
axes[0][0].plot(error_estimate_table["basal noise (a.u.)"].sort_values(), 2.5*exponential(error_estimate_table["basal noise (a.u.)"].sort_values(), 1/240), color="red", lw=lw)
axes[0][0].set_xlim([0,0.04])
sns.histplot(x="peak width", data=error_estimate_table[error_estimate_table["peak width"]<0.5], bins=40, ax=axes[0][1], label="all peaks")
sns.histplot(x="peak width", data=error_estimate_table[(error_estimate_table["peak width"]<0.5) & (error_estimate_table["is_signal"]==True)], bins=40, ax=axes[0][1], color="orange", label="signal only")
#axes[0][1].plot(error_estimate_table["peak width"].sort_values(), utils.gaussian(error_estimate_table["peak width"].sort_values(), 100, peak_width_mean, peak_width_std), color="purple", lw=lw)
axes[0][1].plot(error_estimate_table["peak width"].sort_values(), utils.gaussian(error_estimate_table["peak width"].sort_values(), 100, peak_width_mean, peak_width_std), color="purple", lw=lw)
axes[0][1].plot(error_estimate_table["peak width"].sort_values(), utils.gaussian(error_estimate_table["peak width"].sort_values(), 100, peak_width_mean, peak_width_std/2), color="red", lw=lw)
axes[0][1].set_xlim([0.1,0.5])
plt.legend(frameon=False, fontsize=10)
sns.histplot(x="horizontal error (Da)", data=error_estimate_table[error_estimate_table["horizontal error (Da)"]<1], bins=45, ax=axes[1][0])
sns.histplot(x="horizontal error (Da)", data=error_estimate_table[(error_estimate_table["horizontal error (Da)"]<1) & (error_estimate_table["is_signal"]==True)], bins=45, ax=axes[1][0], color="orange")
#[1][0].plot(error_estimate_table["horizontal error (Da)"].sort_values(), utils.gaussian(error_estimate_table["horizontal error (Da)"].sort_values(), 1e3, 0, 0.1), color="purple", lw=lw)
axes[1][0].plot(error_estimate_table["horizontal error (Da)"].sort_values(), 30*exponential(error_estimate_table["horizontal error (Da)"].sort_values(), 1/20), color="purple", lw=lw)
axes[1][0].plot(error_estimate_table["horizontal error (Da)"].sort_values(), 15*exponential(error_estimate_table["horizontal error (Da)"].sort_values(), 1/40), color="red", lw=lw)
axes[1][0].set_xlim([0,1])
sns.histplot(x="vertical error (rel.)", data=error_estimate_table[error_estimate_table["vertical error (rel.)"]<2.5], bins=50, ax=axes[1][1])
sns.histplot(x="vertical error (rel.)", data=error_estimate_table[(error_estimate_table["vertical error (rel.)"]<2.5) & (error_estimate_table["is_signal"]==True)], bins=50, ax=axes[1][1], color="orange")
#plt.plot(error_estimate_table["vertical error (rel.)"].sort_values(), utils.gaussian(error_estimate_table["vertical error (rel.)"].sort_values(), 300, 1, 0.25), color="purple", lw=lw)
plt.plot(error_estimate_table["vertical error (rel.)"].sort_values(), utils.gaussian(error_estimate_table["vertical error (rel.)"].sort_values(), 100, 1, 0.2), color="purple", lw=lw)
plt.plot(error_estimate_table["vertical error (rel.)"].sort_values(), utils.gaussian(error_estimate_table["vertical error (rel.)"].sort_values(), 100, 1, 0.1), color="red", lw=lw)
axes[1][1].set_xlim([0,2.5])
axes[1][1].set_ylim([-1,300])
axes[1][1].set_yticks([50, 100, 150])
sns.despine()
plt.tight_layout()
plt.show()

###################################################################################################################
###################################################################################################################


###################################################################################################################
########################################### SUPPLEMENTAL FIGURE 4 AND 5 ###########################################
###################################################################################################################
modform_file_name = "complex" # "phospho"
modform_distribution = pd.read_csv("../data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", 
                                   sep=",")
data_simulation = SimulateData(aa_sequence_str, modifications_table)
data_simulation.set_peak_width_mean(peak_width_mean)

data_simulation.reset_noise_error()
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(peak_width_std=peak_width_std)
masses_p, intensities_p = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(peak_width_std=peak_width_std,  horizontal_error_beta=1/20)
masses_ph, intensities_ph = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(peak_width_std=peak_width_std,  horizontal_error_beta=1/20, basal_noise_beta=1/120)
masses_phb, intensities_phb = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(peak_width_std=peak_width_std,  horizontal_error_beta=1/20, basal_noise_beta=1/120,
                          vertical_error_std=0.2)
masses_phbv, intensities_phbv = data_simulation.create_mass_spectrum(modform_distribution)

title = ["no noise or error", "peak width variation", "peak width variation + horizontal error",
         "peak width variation + horizontal error + basal noise",
         "peak width variation + horizontal and vertical error + basal noise"]
fig = plt.figure(figsize=(7,6))
gs = fig.add_gridspec(5, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)
axes[0].plot(masses, intensities, color="k") 
axes[1].plot(masses_p, intensities_p, color="k") 
axes[2].plot(masses_ph, intensities_ph, color="k") 
axes[3].plot(masses_phb, intensities_phb, color="k") 
axes[4].plot(masses_phbv, intensities_phbv, color="k") 
axes[4].set_xlabel("mass (Da)", fontsize=10)
axes[2].set_ylabel("intensity (a.u.)", fontsize=10)
[axes[i].set_title(title[i], fontsize=10, pad=-20) for i in  range(5)]
#[axes[i].set_xlim([43680, 44220]) for i in range(5)] # phospho
#[axes[i].set_ylim([0, 1210]) for i in range(5)] # phospho
[axes[i].set_xlim([43720, 44460]) for i in range(5)] # complex
[axes[i].set_ylim([0, 2350]) for i in range(5)] # complex
fig.tight_layout()
sns.despine()
plt.show()
###################################################################################################################
###################################################################################################################


###################################################################################################################
#################################################### FIGURE 2 #####################################################
###################################################################################################################
modform_file_name = "phospho"
performance_df = pd.read_csv("../output/performance_"+modform_file_name+".csv")

vertical_error_std_list = [0, 0.1, 0.2]
horizontal_error_std_list = [0, 1/40, 1/20]
peak_width_std_list = [0., peak_width_std/2, peak_width_std]
basal_noise_beta_list = [0, 1/240, 1/120]

all_std_combinations = [p for p in itertools.product(*[peak_width_std_list, horizontal_error_std_list, 
                                                       vertical_error_std_list, basal_noise_beta_list])]

basal_noise_peak_width_comb = [p for p in itertools.product(*[basal_noise_beta_list[::-1], peak_width_std_list])]

performance_df["ptm_pattern_acc"]=performance_df["matching_ptm_patterns"]/performance_df["simulated_mass_shifts"]

metric = "ptm_pattern_acc" # "r_score_abundance" # "r_score_mass" "ptm_pattern_acc" 

fig, axn = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(7,6))
cbar_ax = fig.add_axes([.93, 0.3, 0.02, 0.4])
cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
for i, ax in enumerate(axn.flat):
    basal_noise_beta = basal_noise_peak_width_comb[i][0]
    peak_width_std = round(basal_noise_peak_width_comb[i][1],6)
    mask = ((performance_df["peak_width_std"].round(6)==round(peak_width_std,6)) & 
            (performance_df["basal_noise_beta"].round(6)==round(basal_noise_beta,6)))
    pivot_df = performance_df[mask].pivot("vertical_error_std", "horizontal_error_beta", metric)                                                                             
    sns.heatmap(pivot_df, ax=ax,
                cbar=i == 0, cmap=cmap,
                vmin=performance_df[metric].min(), vmax=performance_df[metric].max(),
                cbar_ax=None if i else cbar_ax)
    
    if i==6 or i==7 or i==8: ax.set_xlabel("$\\beta_{horizontal\_error}$")
    else: ax.set_xlabel("")
    if i==0 or i==3 or i==6: ax.set_ylabel("$\sigma_{vertical\_error}$")
    else: ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)    
    ax.invert_yaxis()
    matching_mass_shifts = performance_df[mask]["matching_mass_shifts"].values[i]
    simulated_mass_shifts = performance_df[mask]["simulated_mass_shifts"].values[i]
    ax.set_title(str(matching_mass_shifts)+" out of "+str(simulated_mass_shifts))
       
fig.tight_layout(rect=[0, 0, 0.93, 1])
###################################################################################################################
###################################################################################################################

