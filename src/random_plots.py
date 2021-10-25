# -*- coding: utf-8 -*-
"""
Created on Oct 20 2021

@author: Marjan Faizi
"""

import sys
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mass_spec_data import MassSpecData
import myconfig as config
import utils

plt.style.use("seaborn-pastel")


mass_shifts_df = pd.read_csv("../output/mass_shifts.csv", sep=",")
parameter = pd.read_csv("../output/parameter.csv", sep=",", index_col=[0])
file_names = [file for file in glob.glob(config.path+config.file_name_ending)] 

#########################################################################################################################
########################### Figure 1-3: Raw I2MS data in comparison to the fitted mass shifts ###########################
#########################################################################################################################
for rep in config.replicates:

    file_names_same_replicate = [file for file in file_names if re.search(rep, file)]
    
    output_fig = plt.figure(figsize=(14,7))
    gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
    axes = gs.subplots(sharex=True, sharey=True)
    ylim_max = mass_shifts_df.filter(regex="y_intensities.*"+rep).max().max()
        
    for cond in config.conditions:        
        
        color_of_sample = config.color_palette[cond][0]
        order_in_plot = config.color_palette[cond][1]
        
        calibration_number = parameter.loc["calibration_number", cond+"_"+rep]
        noise_level = parameter.loc["noise_level", cond+"_"+rep]

        sample_name = [file for file in file_names_same_replicate if re.search(cond, file)]    

        if not sample_name:
            print("\nCondition not available.\n")
            sys.exit()
        else:
            sample_name = sample_name[0]

        data = MassSpecData(sample_name)
        axes[order_in_plot].plot(data.masses+calibration_number, data.intensities, label=cond, color=color_of_sample)
        
        masses = mass_shifts_df["masses "+cond+"_"+rep].dropna().values
        intensities = mass_shifts_df["y_intensities "+cond+"_"+rep].dropna().values
        axes[order_in_plot].plot(masses, intensities, '.', color='0.3')

        stddevs = utils.mapping_mass_to_stddev(masses)
        x_gauss_func = np.arange(config.start_mass_range, config.start_mass_range+config.max_mass_shift)
        y_gauss_func = utils.multi_gaussian(x_gauss_func, intensities, masses, stddevs)
        axes[order_in_plot].plot(x_gauss_func, y_gauss_func, color='0.3')

        for avg in mass_shifts_df["average mass"].values:
            axes[order_in_plot].axvline(x=avg, c='0.3', ls='--', lw=0.3, zorder=0)
            axes[order_in_plot].label_outer()

        axes[order_in_plot].axhline(y=noise_level, c='r', lw=0.3)
        axes[order_in_plot].set_xlim((config.start_mass_range, config.start_mass_range+config.max_mass_shift))
        axes[order_in_plot].set_ylim((-10, ylim_max*1.1))
        axes[order_in_plot].yaxis.grid()
        axes[order_in_plot].legend(fontsize=11, loc='upper right')
                
    plt.xlabel('mass (Da)'); plt.ylabel('intensity')
    output_fig.tight_layout()
    plt.savefig('../output/identified_masses_'+rep+'.pdf', dpi=800)
    plt.show()
#########################################################################################################################
#########################################################################################################################



#########################################################################################################################
##################### Figure 4: Compare identified mass shifts across all conditions and replicates #####################
#########################################################################################################################
output_fig = plt.figure(figsize=(14,7))
gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)

for cond in config.conditions:

    color_of_sample = config.color_palette[cond][0]
    order_in_plot = config.color_palette[cond][1]
    x = mass_shifts_df.filter(regex="masses "+cond).mean(axis=1)
    is_nan_mask = np.isnan(x)
    y = mass_shifts_df.filter(regex="rel. abundances "+cond).mean(axis=1)
    xerr = mass_shifts_df.filter(regex="masses "+cond).std(axis=1)
    yerr = mass_shifts_df.filter(regex="rel. abundances "+cond).std(axis=1)
    axes[order_in_plot].errorbar(x[~is_nan_mask], y[~is_nan_mask], yerr=yerr[~is_nan_mask], xerr=xerr[~is_nan_mask], 
                                  color= color_of_sample, marker="o", markersize=3.5, linestyle="none", label=cond)
    axes[order_in_plot].legend(fontsize=11, loc='upper right')
    axes[order_in_plot].grid(False)
    axes[order_in_plot].yaxis.grid()

    for avg in mass_shifts_df["average mass"].values:
        axes[order_in_plot].axvline(x=avg, c='0.3', ls='--', lw=0.2, zorder=0)

plt.xlabel('mass (Da)'); plt.ylabel('relative abundance')
plt.savefig('../output/comparison_all_replicates.pdf', dpi=800)
plt.show()
#########################################################################################################################
#########################################################################################################################




# def keep_reproduced_results(self):
#     keep_masses_indices = np.arange(len(self.identified_masses_df))
#     for cond in config.conditions:
#         missing_masses_count = self.identified_masses_df.filter(regex="masses .*"+cond).isnull().sum(axis=1)
#         keep_masses_indices = np.intersect1d(keep_masses_indices, self.identified_masses_df[missing_masses_count>=2].index)
        
#     self.identified_masses_df.drop(index=keep_masses_indices, inplace=True)
