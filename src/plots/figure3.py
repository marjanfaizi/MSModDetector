#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 5 2022

@author: Marjan Faizi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from scipy import stats


from simulate_data import SimulateData
import config


### set figure style
sns.set()
sns.set_style("white")
sns.set_context("paper")



###################################################################################################################
########################################### SUPPLEMENTAL FIGURE 4 AND 5 ###########################################
###################################################################################################################
error_estimate_table = pd.read_csv("../output/error_noise_distribution_table.csv")
basal_noise = error_estimate_table["basal noise (a.u.)"].values
horizontal_error = error_estimate_table[(error_estimate_table["horizontal error (Da)"]<0.28) &
                                        (error_estimate_table["horizontal error (Da)"]>-0.28) &
                                        (error_estimate_table["is_signal"]==True)]["horizontal error (Da)"].values
vertical_error = error_estimate_table[error_estimate_table["is_signal"]==True]["vertical error (rel.)"].values

vertical_error_par = list(stats.beta.fit(vertical_error))
horizontal_error_par = list(stats.beta.fit(horizontal_error[(horizontal_error>-0.2) & (horizontal_error<0.2)]))
basal_noise_par = list(stats.beta.fit(basal_noise))

modform_file_name = "complex" # "complex" "phospho"
modform_distribution = pd.read_csv("../data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")
data_simulation = SimulateData(aa_sequence_str, modifications_table)
data_simulation.set_peak_width_mode(0.25)

data_simulation.reset_noise_error()
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.reset_noise_error()
data_simulation.add_noise(horizontal_error_par=horizontal_error_par, basal_noise_par=basal_noise_par, 
                          vertical_error_par=vertical_error_par)
masses_error, intensities_error = data_simulation.create_mass_spectrum(modform_distribution)

# phospho
title = ["Theoretical phosphorylation patterns", "Horizontal and vertical error + basal noise"]

fig = plt.figure(figsize=(7, 2.2))
gs = fig.add_gridspec(2, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)
axes[0].plot(masses, intensities, color="k") 
axes[1].plot(masses_error, intensities_error, color="k") 
axes[1].set_xlabel("mass (Da)", fontsize=10)
axes[1].set_ylabel("intensity (a.u.)", fontsize=10)
axes[0].set_title(title[0], fontsize=10, pad=-5)
axes[1].set_title(title[1], fontsize=10, pad=-15)
[axes[i].set_xlim([43600, 44220]) for i in range(2)] # phospho
[axes[i].set_ylim([0, 1210]) for i in range(2)] # phospho
fig.tight_layout()
sns.despine()
plt.show()

# overlap
fig = plt.figure(figsize=(7, 1.8))
plt.plot(masses_error, intensities_error, color="k") 
plt.plot(modform_distribution["mass"]+config.unmodified_species_mass, modform_distribution["intensity"], ".r", markersize=2) 
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("intensity (a.u.)", fontsize=10)
#plt.xlim([43625, 44230])
plt.xlim([43750, 44390])
plt.ylim([-50, 2055])
#plt.ylim([-50, 1255])
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
#################################################### FIGURE 3 #####################################################
###################################################################################################################
modform_file_name = "complex"
performance_df = pd.read_csv("../output/performance_"+modform_file_name+".csv")

vertical_error = [0, 1]
horizontal_error = [0, 1]
basal_noise = [1, 0]

vertical_horizontal_comb = [p for p in itertools.product(*[vertical_error[::-1], horizontal_error])]

# matching_mass_shifts, r_score_abundance, matching_ptm_patterns # mass_shift_deviation
metric = "r_score_abundance" 

fig, axn = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(1.4, 2.2))
cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
for i, ax in enumerate(axn.flat):
    pivot_df = performance_df[performance_df["basal_noise"]==basal_noise[i]].pivot("vertical_error", "horizontal_error", metric)      
                                                                         
    sns.heatmap(pivot_df, ax=ax, annot=True, cmap=cmap,cbar=None, fmt=".2",
                vmin=0, vmax=1, annot_kws={"size": 9}, cbar_ax=None)

    if i==0 or i==1: ax.set_ylabel("vertical error", fontsize=9)

    if i==1: ax.set_xlabel("horizontal error", fontsize=9)
    else: ax.set_xlabel("")
    ax.set_xticklabels(["no","yes"], rotation=45)
    ax.set_yticklabels(["no","yes"], rotation=90)
    ax.invert_yaxis() 

fig.tight_layout()
###################################################################################################################
###################################################################################################################

