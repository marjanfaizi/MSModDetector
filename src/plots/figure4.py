#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 5 2022

@author: Marjan Faizi
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
from scipy import stats

sys.path.append("../../")
sys.path.append("../simulated_data/")

from simulate_data import SimulateData
import config
import utils


### set figure style
sns.set()
sns.set_style("white")
sns.set_context("paper")


### required input
error_estimate_table = pd.read_csv("../../output/error_noise_distribution_table.csv")
basal_noise = error_estimate_table["basal_noise"].values
horizontal_error = error_estimate_table[(error_estimate_table["horizontal_error"]<0.3) &
                                        (error_estimate_table["horizontal_error"]>-0.3) &
                                        (error_estimate_table["is_signal"]==True)]["horizontal_error"].values
vertical_error = error_estimate_table[(error_estimate_table["vertical_error"]<0.25) &
                                        (error_estimate_table["vertical_error"]>-0.25) &
                                        (error_estimate_table["is_signal"]==True)]["vertical_error"].values
vertical_error_par = list(stats.beta.fit(vertical_error))
horizontal_error_par = list(stats.beta.fit(horizontal_error[(horizontal_error>-0.2) & (horizontal_error<0.2)]))
basal_noise_par = list(stats.beta.fit(basal_noise))

modifications_table = pd.read_csv("../../"+config.modfication_file_name, sep=";")
modifications_table["unimod_id"] = modifications_table["unimod_id"].astype("Int64")

protein_entries = utils.read_fasta("../../"+config.fasta_file_name)
protein_sequence = list(protein_entries.values())[0]    

unmodified_species_mass, stddev_isotope_distribution = utils.isotope_distribution_fit_par(protein_sequence, 100)


### figure A: simulated spectrum phosphorylation patterns
modform_file_name = "phospho"
modform_distribution = pd.read_csv("../simulated_data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")

data_simulation = SimulateData(protein_sequence, modifications_table)
data_simulation.reset_noise_error()
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(horizontal_error_par=horizontal_error_par, basal_noise_par=basal_noise_par, 
                          vertical_error_par=vertical_error_par)
masses_error_noise, intensities_error_noise = data_simulation.create_mass_spectrum(modform_distribution)

title = ["Theoretical phosphorylation patterns", "Horizontal and vertical error + basal noise"]

fig = plt.figure(figsize=(7, 2.2))
gs = fig.add_gridspec(2, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)
axes[0].plot(masses, intensities, color="0.3") 
axes[1].plot(masses_error_noise, intensities_error_noise, color="0.3") 
axes[1].set_xlabel("mass (Da)")
axes[1].set_ylabel("intensity (a.u.)")
axes[0].set_title(title[0], pad=-5)
axes[1].set_title(title[1], pad=-15)
[axes[i].set_xlim([43600, 44220]) for i in range(2)]
[axes[i].set_ylim([0, 1210]) for i in range(2)] 
fig.tight_layout()
sns.despine()
plt.show()


### figure B: performance evaluation on phophospho patterns
modform_file_name = "phospho"
#performance_df = pd.read_csv("../../output/performance_"+modform_file_name+"_50_simulations.csv")
performance_df = pd.read_csv("../../output/performance_"+modform_file_name+".csv")

vertical_error = [0, 1]
horizontal_error = [0, 1]
basal_noise = [1, 0]

vertical_horizontal_comb = [p for p in itertools.product(*[vertical_error[::-1], horizontal_error])]

metric = ["matching_mass_shifts", "r_score_abundance", "matching_ptm_patterns_min_ptm", 
          "matching_ptm_patterns_min_err", "matching_ptm_patterns_min_both"] 

vmin = [0, 0.99, 0, 0, 0]
vmax = [modform_distribution.shape[0], 1, modform_distribution.shape[0], modform_distribution.shape[0], modform_distribution.shape[0]]
fmt = [False, True, False, False, False]
cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

for ix, m in enumerate(metric):
    fig, axn = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(1.4, 2.2))
    
    for i, ax in enumerate(axn.flat):
        pivot_df = performance_df[performance_df["basal_noise"]==basal_noise[i]].pivot("vertical_error", "horizontal_error", m)      
        
        if fmt[ix]:
            sns.heatmap(pivot_df, ax=ax, annot=True, cmap=cmap,cbar=None, fmt="0.3",
                        vmin=vmin[ix], vmax=vmax[ix], annot_kws={"size": 9}, cbar_ax=None)
        else:
            sns.heatmap(pivot_df, ax=ax, annot=True, cmap=cmap,cbar=None,
                        vmin=vmin[ix], vmax=vmax[ix], annot_kws={"size": 9}, cbar_ax=None)
    
        if i==0 or i==1: ax.set_ylabel("vertical error", fontsize=9)
    
        if i==1: ax.set_xlabel("horizontal error", fontsize=9)
        else: ax.set_xlabel("")
        ax.set_xticklabels(["no","yes"], rotation=45)
        ax.set_yticklabels(["no","yes"], rotation=90)
        ax.invert_yaxis() 
    
    fig.tight_layout()


### figure C: overlapping isotopic distribution
modform_file_name = "overlap"
modform_distribution = pd.read_csv("../simulated_data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")

data_simulation = SimulateData(protein_sequence, modifications_table)
data_simulation.reset_noise_error()
data_simulation.add_noise(horizontal_error_par=horizontal_error_par, basal_noise_par=basal_noise_par, 
                          vertical_error_par=vertical_error_par)
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

fig = plt.figure(figsize=(7, 1.5))
plt.plot(masses, intensities, color="0.3") 
plt.plot(modform_distribution["mass"]+unmodified_species_mass, modform_distribution["intensity"], ".r", markersize=2) 
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim([43640, 44180])
plt.ylim([-50, 1100])
fig.tight_layout()
sns.despine()
plt.show()


# overlaps of 1 Da and 2 Da will not be seperated, to close within specified mass tolerance
# overlap to modform in table mapping
overlaps = {"4 Da": [4,5], "6 Da": [6,7], "8 Da": [8,9], "10 Da": [10,11], "12 Da": [12,13], 
            "14 Da": [14,15], "16 Da": [16,17]}

# "arr_0": true_mass_shift, "arr_1": chi_sqaure_score, "arr_2": mass_shift_deviation, 
# "arr_3": ptm_patterns, "arr_4": ptm_patterns_top3, "arr_5": ptm_patterns_top5, "arr_6": ptm_patterns_top10
npzfile = np.load("../../output/evaluated_overlap_data_50_simulations.npz")

amount_simulations = npzfile["arr_0"].shape[0]

for key in overlaps:
    first_modform_ix = overlaps[key][0]
    second_modform_ix = overlaps[key][1]
    detected_mass_shift_in_overlap = npzfile["arr_0"][:,first_modform_ix]+npzfile["arr_0"][:,second_modform_ix]
    seperated = sum(1 for i in detected_mass_shift_in_overlap if i == 2)
    print(key+": ", 100*seperated/amount_simulations, "% seperated")





