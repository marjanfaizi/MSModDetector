#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 2022

@author: Marjan Faizi
"""

import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
sys.path.append("../../")

from mass_spec_data import MassSpecData
import config
import utils


### set figure style
sns.set()
sns.set_style("white")
sns.set_context("paper")
sns.set_style("ticks")


### required input 
file_names = [file for file in glob.glob("../../"+config.file_names)] 
mass_shifts_df = pd.read_csv("../../output/mass_shifts.csv", sep=",")
ptm_patterns_df = pd.read_csv("../../output/ptm_patterns_table.csv", sep=",")
parameter = pd.read_csv("../../output/parameter.csv", sep=",", index_col=[0])
cond_mapping = {"nutlin_only": "Nutlin-3a", "uv_7hr": "UV"}
protein_entries = utils.read_fasta("../../"+config.fasta_file_name)
protein_sequence = list(protein_entries.values())[0]
unmodified_species_mass, stddev_isotope_distribution = utils.isotope_distribution_fit_par(protein_sequence, 100)


### figure A
flip_spectrum = [1,-1]
output_fig = plt.figure(figsize=(7, 2.8))
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
        data = MassSpecData()
        data.add_raw_spectrum(sample_name)
                
        masses = mass_shifts_df["masses "+cond+"_"+rep].dropna().values
        intensities = mass_shifts_df["raw intensities "+cond+"_"+rep].dropna().values

        x_gauss_func = np.arange(config.mass_range_start, config.mass_range_end)
        y_gauss_func = utils.multi_gaussian(x_gauss_func, intensities, masses, stddev_isotope_distribution)
        
        axes[order_in_plot].plot(data.raw_spectrum[:,0], flip_spectrum[ix]*data.raw_spectrum[:,1]/(rescaling_factor*total_protein_abundance), label=cond_mapping[cond], color=color_of_sample)
        axes[order_in_plot].plot(masses, flip_spectrum[ix]*intensities/total_protein_abundance, '.', color='0.3')
        axes[order_in_plot].plot(x_gauss_func,flip_spectrum[ix]*y_gauss_func/total_protein_abundance, color='0.3')
        axes[order_in_plot].axhline(y=flip_spectrum[ix]*noise_level/total_protein_abundance, c='r', lw=0.3)
        if flip_spectrum[ix] > 0: axes[order_in_plot].legend(loc='upper right')

ylim_max = mass_shifts_df.filter(regex="raw intensities.*").max().max()      
plt.xlim((config.mass_range_start, config.mass_range_end))
plt.ylim((-(ylim_max/total_protein_abundance)*1.4, (ylim_max/total_protein_abundance)*1.4))
plt.xlabel("mass (Da)"); plt.ylabel("rel. abundance") 
output_fig.tight_layout()
plt.show()


### figure B
col_name_nutlin = [col for col in mass_shifts_df if col.startswith('rel. abundances nutlin')]
nutlin_df = pd.melt(mass_shifts_df, id_vars="mass shift", value_vars=col_name_nutlin, 
                    value_name="rel. abundance").rename(columns={"variable": "condition"})
nutlin_df["condition"] = "Nutlin-3a"

col_name_uv = [col for col in mass_shifts_df if col.startswith('rel. abundances uv')]
uv_df = pd.melt(mass_shifts_df, id_vars="mass shift", value_vars=col_name_uv, 
                    value_name="rel. abundance").rename(columns={"variable": "condition"})
uv_df["condition"] = "UV"

all_condition_df = pd.concat([nutlin_df, uv_df])
all_condition_df["mass shift"] = all_condition_df["mass shift"].astype(int)
all_condition_df.rename(columns={"mass shift": "mass shift (Da)"}, inplace=True)

fig = plt.figure(figsize=(7, 2))
sns.barplot(x="mass shift (Da)", y="rel. abundance", hue="condition", data=all_condition_df,
            palette=["skyblue","mediumpurple"], errwidth=1)
plt.xticks(rotation=60)
sns.despine()
fig.tight_layout()
plt.gca().legend().set_title("")
plt.legend(loc="upper right")
plt.show()


### supplement table information
# max bin size for each mass shift
mass_shifts_df.filter(regex="masses ").max(axis=1)-mass_shifts_df.filter(regex="masses ").min(axis=1)

# mass error for each mass shift 
ptm_patterns_df.groupby("mass shift").first()["mass error (Da)"]
