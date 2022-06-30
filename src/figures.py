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
import glob
import re
from scipy import stats
from scipy.optimize import minimize

from simulate_data import SimulateData
from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
import config
import utils

sns.set()
sns.set_style("white")
sns.set_context("paper")

###################################################################################################################
############################################## INPUT AND TOY EXAMPLE ##############################################
###################################################################################################################
file_name = "../data/raw_data/P04637/xray_7hr_rep5.mzml"
rep = "rep5"
cond = "xray_7hr"

data = MassSpecData(file_name)
data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)  

protein_entries = utils.read_fasta(config.fasta_file_name)
aa_sequence_str = list(protein_entries.values())[0]
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


protein_entries = utils.read_fasta(config.fasta_file_name)
mean_p53, stddev_p53, amplitude__p53 = utils.isotope_distribution_fit_par(list(protein_entries.values())[0], 100)

protein_sequence = AASequence.fromString(list(protein_entries.values())[0])
distribution = utils.get_theoretical_isotope_distribution(protein_sequence, 100)
mass_grid, intensities = utils.fine_grain_isotope_distribution(distribution, 0.2, 0.02)



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

############## For poster only
error_estimate_table = pd.read_csv("../output/error_noise_distribution_table.csv")
basal_noise = error_estimate_table["basal noise (a.u.)"].values
peak_width = error_estimate_table[(error_estimate_table["peak width"]<0.5) & 
                                  (error_estimate_table["is_signal"]==True)]["peak width"].values
horizontal_error = error_estimate_table[error_estimate_table["is_signal"]==True]["horizontal error (Da)"].values
vertical_error = error_estimate_table[(error_estimate_table["vertical error (rel.)"]<2.5) &
                                      (error_estimate_table["is_signal"]==True)]["vertical error (rel.)"].values

vertical_error_par = list(stats.beta.fit(vertical_error))
horizontal_error_par = list(stats.expon.fit(horizontal_error))
peak_width_par = list(stats.beta.fit(peak_width))
basal_noise_par = list(stats.beta.fit(basal_noise))

distribution = utils.get_theoretical_isotope_distribution(protein_sequence, 100)
scaling_factor = distribution[:,1].max()

basal_noise = (basal_noise_par[3]*np.random.beta(*basal_noise_par[:2], size=distribution.shape[0]))+basal_noise_par[2]
distribution[:,1] += 4*basal_noise*scaling_factor

vertical_error = (vertical_error_par[3]*np.random.beta(*vertical_error_par[:2], size=distribution.shape[0]))+vertical_error_par[2]
distribution[:,1] *= vertical_error

horizontal_error = np.random.exponential(horizontal_error_par[1], size=distribution.shape[0])+horizontal_error_par[0]
distribution[:,0] += horizontal_error

mass_grid, intensities = utils.fine_grain_isotope_distribution(distribution, 0.2, 0.02)


plt.figure(figsize=(5, 2.5))
plt.plot(mass_grid, intensities, '-', color="0.3", label="isotopic distribution of p53")
plt.plot(mass_grid, utils.gaussian(mass_grid, intensities.max(), mean_p53, stddev_p53)/1.5, 'r-', 
         label="Gaussian fit") 
plt.plot(mean_p53, intensities.max()/1.5, "r.")
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("intensity (a.u.)", fontsize=10)
plt.xlim((43630, 43680))
#plt.legend(frameon=False, fontsize=10)
plt.tight_layout()
sns.despine()
plt.show()
###################################################################################################################
###################################################################################################################


###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 3 ##############################################
###################################################################################################################
error_estimate_table = pd.read_csv("../output/error_noise_distribution_table_06_29_22.csv")

spacing_between_peaks = 1.003355
lw = 1.3
binsize = 30
fig, axes = plt.subplots(2, 2, figsize=(6,4.5))
# basal noise
basal_noise = error_estimate_table["basal noise (a.u.)"].values
axes[0][0].hist(basal_noise, bins=binsize, density=True, alpha=0.5)
axes[0][0].set_xlabel("basal noise (a.u)")
axes[0][0].set_ylabel("density")
x = np.linspace(-1e-3, basal_noise.max(), len(basal_noise))
a, b, loc, scale = stats.beta.fit(basal_noise)  
pdf_beta = stats.beta.pdf(x, a, b, loc, scale)  
axes[0][0].plot(x, pdf_beta, color="purple", lw=lw)
# peak width variation
peak_width = error_estimate_table[(error_estimate_table["peak width"]<0.45) & 
                                  (error_estimate_table["is_signal"]==True)]["peak width"].values
axes[0][1].hist(peak_width, bins=binsize, density=True, alpha=0.5)
axes[0][1].set_xlabel("peak width")
axes[0][1].set_ylabel("density")
x = np.linspace(peak_width.min(), peak_width.max(), len(peak_width))
a, b, loc, scale = stats.beta.fit(peak_width)  
pdf_beta = stats.beta.pdf(x, a, b, loc, scale)  
axes[0][1].plot(x, pdf_beta, color="purple", lw=lw)
func = lambda x: -stats.beta.pdf(x, a, b, loc, scale)  
peak_width_mode = minimize(func, 0.2).x
# horizontal error
horizontal_error = error_estimate_table[(error_estimate_table["horizontal error (Da)"]<0.3) &
                                        (error_estimate_table["horizontal error (Da)"]>-0.3) &
                                        (error_estimate_table["is_signal"]==True)]["horizontal error (Da)"].values
axes[1][0].hist(horizontal_error, bins=binsize, density=True, alpha=0.5)
axes[1][0].set_xlabel("horizontal error (Da)")
axes[1][0].set_ylabel("density")
x = np.linspace(-0.3, horizontal_error.max(), len(horizontal_error))
#loc, scale = stats.expon.fit(horizontal_error[(horizontal_error<0.5) & (horizontal_error>-0.5)])  
a, b, loc, scale = stats.beta.fit(horizontal_error)  
pdf_beta = stats.beta.pdf(x, a, b, loc, scale) 
axes[1][0].plot(x, pdf_beta, color="purple", lw=lw)
# vertical error
vertical_error = error_estimate_table[(error_estimate_table["vertical error (rel.)"]<5) &
                                      (error_estimate_table["is_signal"]==True)]["vertical error (rel.)"].values
axes[1][1].hist(vertical_error, bins=binsize, density=True, alpha=0.5)
axes[1][1].set_xlabel("vertical error (rel.)")
axes[1][1].set_ylabel("density")
x = np.linspace(-0.1, vertical_error.max(), len(vertical_error))
a, b, loc, scale = stats.beta.fit(vertical_error[(vertical_error>0.5) & (vertical_error<2)])  
pdf_beta = stats.beta.pdf(x, a, b, loc, scale)  
axes[1][1].plot(x, pdf_beta, color="purple", lw=lw)
# plot layout
sns.despine()
plt.tight_layout()
plt.show()
###################################################################################################################
###################################################################################################################



###################################################################################################################
########################################### SUPPLEMENTAL FIGURE 4 AND 5 ###########################################
###################################################################################################################
basal_noise = error_estimate_table["basal noise (a.u.)"].values
peak_width = error_estimate_table[(error_estimate_table["peak width"]<0.5) & 
                                  (error_estimate_table["is_signal"]==True)]["peak width"].values
horizontal_error = error_estimate_table[error_estimate_table["is_signal"]==True]["horizontal error (Da)"].values
vertical_error = error_estimate_table[(error_estimate_table["vertical error (rel.)"]<2.5) &
                                      (error_estimate_table["is_signal"]==True)]["vertical error (rel.)"].values

vertical_error_par = list(stats.beta.fit(vertical_error[(vertical_error>0.5) & (vertical_error<2)]))
horizontal_error_par = list(stats.beta.fit(horizontal_error))
peak_width_par = list(stats.beta.fit(peak_width))
basal_noise_par = list(stats.beta.fit(basal_noise))

modform_file_name = "phospho" # "complex"
modform_distribution = pd.read_csv("../data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")
data_simulation = SimulateData(aa_sequence_str, modifications_table)
data_simulation.set_peak_width_mode(peak_width_mode[0])

data_simulation.reset_noise_error()
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(peak_width_par=peak_width_par)
masses_p, intensities_p = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(peak_width_par=peak_width_par,  horizontal_error_par=horizontal_error_par)
masses_ph, intensities_ph = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(peak_width_par=peak_width_par,  horizontal_error_par=horizontal_error_par, 
                          basal_noise_par=basal_noise_par)
masses_phb, intensities_phb = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(peak_width_par=peak_width_par,  horizontal_error_par=horizontal_error_par, 
                          basal_noise_par=basal_noise_par, vertical_error_par=vertical_error_par)
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
[axes[i].set_xlim([43600, 44220]) for i in range(5)] # phospho
[axes[i].set_ylim([0, 1210]) for i in range(5)] # phospho
#[axes[i].set_xlim([43600, 44460]) for i in range(5)] # complex
#[axes[i].set_ylim([0, 2350]) for i in range(5)] # complex
fig.tight_layout()
sns.despine()
plt.show()


sns.set_context("poster", font_scale=0.8)

fig = plt.figure(figsize=(13.5,3.5))
gs = fig.add_gridspec(2, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)
axes[0].plot(masses, intensities, color="0.3") 
axes[1].plot(masses_phbv, intensities_phbv, color="0.3") 
axes[1].set_xlabel("mass (Da)")#, fontsize=10)
axes[1].set_ylabel("intensity (a.u.)")#, fontsize=10)
[axes[i].set_xlim([43615, 44180]) for i in range(2)] # phospho
[axes[i].set_ylim([0, 1510]) for i in range(2)] # phospho
#[axes[i].set_xlim([43600, 44460]) for i in range(2)] # complex
#[axes[i].set_ylim([0, 2450]) for i in range(2)] # complex
fig.tight_layout()
sns.despine()
plt.show()
###################################################################################################################
###################################################################################################################



###################################################################################################################
#################################################### FIGURE 2 #####################################################
###################################################################################################################
sns.set_style("ticks")

cond_mapping = {"nutlin_only": "Nutlin-3a", "xray_2hr": "X-ray (2hr)", "xray_7hr": "X-ray (7hr)",
                "xray-nutlin": "X-ray + Nutlin-3a", "uv_7hr": "UV"}

mass_shifts_df = pd.read_csv("../output/mass_shifts_all_possibilities.csv", sep=",")
ptm_patterns_df = pd.read_csv("../output/ptm_patterns_table_all_possibilities.csv", sep=",")
parameter = pd.read_csv("../output/parameter_all_possibilities.csv", sep=",", index_col=[0])
file_names = [file for file in glob.glob(config.file_names)] 

flip_spectrum = [1,-1]

output_fig = plt.figure(figsize=(7, 4))
gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)

for cond in config.conditions:  
    color_of_sample = config.color_palette[cond][0]
    order_in_plot = config.color_palette[cond][1]
    
    for ix, rep in enumerate(config.replicates):
        file_names_same_replicate = [file for file in file_names if re.search(rep, file)]
        
        noise_level = parameter.loc["noise_level", cond+"_"+rep]
        rescaling_factor = parameter.loc["rescaling_factor", cond+"_"+rep]
        total_protein_abundance = parameter.loc["total_protein_abundance", cond+"_"+rep]

        sample_name = [file for file in file_names_same_replicate if re.search(cond, file)][0]
        data = MassSpecData(sample_name)
                
        masses = mass_shifts_df["masses "+cond+"_"+rep].dropna().values
        intensities = mass_shifts_df["raw intensities "+cond+"_"+rep].dropna().values

        x_gauss_func = np.arange(config.mass_start_range, config.mass_end_range)
        y_gauss_func = utils.multi_gaussian(x_gauss_func, intensities, masses, config.stddev_isotope_distribution)
        
        """
        axes[order_in_plot].plot(data.masses, flip_spectrum[ix]*data.intensities/rescaling_factor, label=cond, color=color_of_sample)
        axes[order_in_plot].plot(masses, flip_spectrum[ix]*intensities, '.', color='0.3')
        axes[order_in_plot].plot(x_gauss_func,flip_spectrum[ix]* y_gauss_func, color='0.3')
        axes[order_in_plot].axhline(y=flip_spectrum[ix]*noise_level, c='r', lw=0.3)
        """
        axes[order_in_plot].plot(data.masses, flip_spectrum[ix]*data.intensities/(rescaling_factor*total_protein_abundance), label=cond_mapping[cond], color=color_of_sample)
        axes[order_in_plot].plot(masses, flip_spectrum[ix]*intensities/total_protein_abundance, '.', color='0.3')
        axes[order_in_plot].plot(x_gauss_func,flip_spectrum[ix]*y_gauss_func/total_protein_abundance, color='0.3')
        #axes[order_in_plot].axhline(y=flip_spectrum[ix]*noise_level/total_protein_abundance, c='r', lw=0.3)

        if flip_spectrum[ix] > 0: axes[order_in_plot].legend(loc='upper right')
        #axes[order_in_plot].yaxis.grid(visible=True, which='major', color='0.3', linestyle='-')

ylim_max = mass_shifts_df.filter(regex="raw intensities.*").max().max()      
plt.xlim((config.mass_start_range, config.mass_end_range))
#plt.ylim((-ylim_max*1.1, ylim_max*1.1))
plt.ylim((-(ylim_max/total_protein_abundance)*1.5, (ylim_max/total_protein_abundance)*1.5))

plt.xlabel("mass (Da)"); plt.ylabel("rel. abundance") # plt.ylabel("rel. intensity")
output_fig.tight_layout()
plt.show()


mass_error_nutlin = abs(mass_shifts_df["masses nutlin_only_rep5"] - mass_shifts_df["masses nutlin_only_rep6"])
mass_error_uv = abs(mass_shifts_df["masses uv_7hr_rep5"] - mass_shifts_df["masses uv_7hr_rep6"])
mass_error_all =  mass_shifts_df.filter(regex="masses ").max(axis=1).values - \
                  mass_shifts_df.filter(regex="masses ").min(axis=1).values



fig = plt.figure(figsize=(7, 1.5))
plt.plot(mass_shifts_df["average mass"]-config.unmodified_species_mass, mass_error_nutlin, ".", color="skyblue", label="Nutlin-3a")
plt.plot(mass_shifts_df["average mass"]-config.unmodified_species_mass, mass_error_uv, ".", color="mediumpurple", label="UV")
plt.plot(mass_shifts_df["average mass"]-config.unmodified_species_mass, mass_error_all, ".", color="0.3", label="All conditions")
plt.xlim((config.mass_start_range-config.unmodified_species_mass, config.mass_end_range-config.unmodified_species_mass))
plt.legend()
plt.ylabel("mass error [Da]")
plt.xlabel("mass shift [$\Delta$ Da]")
sns.despine()
fig.tight_layout()
plt.show()


ptm_patterns_df.groupby("mass shift").size()

###################################################################################################################
###################################################################################################################




###################################################################################################################
#################################################### FIGURE 3 #####################################################
###################################################################################################################
modform_file_name = "phospho_acetyl"
performance_df = pd.read_csv("../output/performance_"+modform_file_name+".csv")

vertical_error = [0, 1]
horizontal_error = [0, 1]
peak_width = [0, 1]
basal_noise = [0, 1]

all_combinations = [p for p in itertools.product(*[peak_width, horizontal_error, 
                                                   vertical_error, basal_noise])]

#basal_noise_peak_width_comb = [p for p in itertools.product(*[basal_noise[::-1], peak_width])]
vertical_horizontal_comb = [p for p in itertools.product(*[vertical_error[::-1], horizontal_error])]


metric = "matching_mass_shifts" # matching_mass_shifts, r_score_abundance, matching_ptm_patterns # mass_shift_deviation

fig, axn = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(2.5,2.5))
#cbar_ax = fig.add_axes([.93, 0.3, 0.02, 0.4])
cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
for i, ax in enumerate(axn.flat):
    """  
    basal_noise = basal_noise_peak_width_comb[i][0]
    peak_width = basal_noise_peak_width_comb[i][1]
    mask = ((performance_df["peak_width_variation"]==peak_width) & 
            (performance_df["basal_noise"]==basal_noise))
    pivot_df = performance_df[mask].pivot("vertical_error", "horizontal_error", metric)
    """
    vertical_error = vertical_horizontal_comb[i][0]
    horizontal_error = vertical_horizontal_comb[i][1]
    mask = ((performance_df["horizontal_error"]==horizontal_error) & 
            (performance_df["vertical_error"]==vertical_error))
    pivot_df = performance_df[mask].pivot("basal_noise", "peak_width_variation", metric)      
                                                                         
    sns.heatmap(pivot_df, ax=ax, annot=True, cmap=cmap,cbar=None, annot_kws={"size": 10},
                vmin=0, vmax=7,
                #cbar=i == 0, cmap=cmap,
                #vmin=performance_df[metric].min(), vmax=performance_df[metric].max(),
                cbar_ax=None) #if i else cbar_ax)
    
#    if i==2 or i==3: ax.set_xlabel("$error_{horizontal}$")
    if i==2 or i==3: ax.set_xlabel("peak width")
    else: ax.set_xlabel("")
#    if i==0 or i==2: ax.set_ylabel("$error_{vertical}$")
    if i==0 or i==2: ax.set_ylabel("basal noise")
    else: ax.set_ylabel("")
    ax.set_xticklabels(["no","yes"], rotation=45)
    ax.set_yticklabels(["no","yes"], rotation=90)
    ax.invert_yaxis() 
     
fig.tight_layout()#(rect=[0, 0, 0.93, 1])
###################################################################################################################
###################################################################################################################



###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 6 ##############################################
###################################################################################################################
"""
20,376 proteins from the human proteome were downloaded from uniprot on the 23rd of May 2022. Isoforms are not 
included and only reviewed proteins were downloaded. Isotopic distribtions of proteins with a mass between 
~10kDa and ~100kDa are well suited to be approximated by Gaussian distributions.
"""

sns.set_style("ticks")

from pyopenms import AASequence
proteome = utils.read_fasta("../data/fasta_files/human_proteome_reviewed_23_05_22.fasta")
results_df = pd.read_csv("../output/fit_gaussian_to_proteome.csv")

plt.figure(figsize=(4, 3))
plt.plot(results_df["monoisotopic_mass"], results_df["pvalue"], '.', color="0.3")
plt.xlabel("monoisotopic mass (Da)", fontsize=10)
plt.ylabel("p-value", fontsize=10)
plt.tight_layout()
sns.despine()
plt.show()

lb = [1000, 3000, 5000, 7000, 9000, 15000, 30000, 50000, 70000, 95000, 120000, 140000, 160000, 180000, 200000, 400000]
ub = [2000, 4000, 6000, 8000, 10000, 20000, 40000, 60000, 80000, 105000, 130000, 150000, 170000, 190000, 210000, 500000]

fig, axn = plt.subplots(4, 4, sharey=True, figsize=(7, 8))
for i, ax in enumerate(axn.flat):
    protein_id = results_df.query("(monoisotopic_mass>"+str(lb[i])+" & monoisotopic_mass<"+str(ub[i])+")").sample(n=1).protein_id.item()
    distribution = utils.get_theoretical_isotope_distribution(AASequence.fromString(proteome[protein_id]), 100)
    mass_grid, intensities = utils.fine_grain_isotope_distribution(distribution, 0.2, 0.02)
    intensities_norm = intensities/intensities.max()
    non_zero_intensity_ix = np.where(intensities > 1e-5)[0]
    ax.plot(mass_grid, intensities_norm, '-', color="0.3", lw=0.7)
    ax.plot(mass_grid, utils.gaussian(mass_grid, results_df[results_df["protein_id"]==protein_id]["amplitude"].item()/intensities.max(), 
                                      results_df[results_df["protein_id"]==protein_id]["mean"].item(), 
                                      results_df[results_df["protein_id"]==protein_id]["standard_deviation"].item()), 'r-')
    
    ax.set_xlim([mass_grid[non_zero_intensity_ix].min(), mass_grid[non_zero_intensity_ix].max()])
    if i>=12 and i<=15: ax.set_xlabel("mass (Da)", fontsize=10)
    else: ax.set_xlabel("")
    if  i==0 or i==4 or i==8 or i==12: ax.set_ylabel("intensity (a.u.)", fontsize=10)
    else: ax.set_ylabel("")

fig.tight_layout()
sns.despine()
plt.show()

###################################################################################################################
###################################################################################################################





