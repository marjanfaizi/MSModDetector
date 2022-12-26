#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 2022

@author: Marjan Faizi
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../../src/")

from mass_spec_data import MassSpecData
import utils


### set figure style
sns.set()
sns.set_style("white")
sns.set_context("paper")
sns.set_style("ticks")


### required input 
mass_range_start = 43750.0
mass_range_end = 44520.0
fasta_file_name = "P04637.fasta"
metadata = pd.read_csv("../../raw_data/metadata.csv")
file_names = ["../../raw_data/"+f for f in metadata.filename.tolist()]
mass_shifts_df = pd.read_csv("../../output/mass_shifts.csv", sep=",")
ptm_patterns_df = pd.read_csv("../../output/ptm_patterns_table.csv", sep=",")
parameter = pd.read_csv("../../output/parameter.csv", sep=",", index_col=[0])
cond_mapping = {"nutlin": "Nutlin-3a", "uv": "UV", "xray_2hr": "X-ray (2hr)", 
                "xray_7hr": "X-ray (7hr)"}
protein_entries = utils.read_fasta("../../fasta_files/"+fasta_file_name)
protein_sequence = list(protein_entries.values())[0]
unmodified_species_mass, stddev_isotope_distribution = utils.isotope_distribution_fit_par(protein_sequence, 100)


### figure A
number_of_conditions = len(metadata.condition.unique().tolist())
color_palette = {"nutlin": "skyblue", "uv": "mediumpurple"}
flip_spectrum = [1, -1, 1, -1]
order_in_plot = [0, 0, 1, 1]
output_fig = plt.figure(figsize=(7, 2.8))
gs = output_fig.add_gridspec(number_of_conditions, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)

for ix, sample_name in enumerate(file_names):
    cond = metadata[metadata.filename == sample_name.split("/")[-1]].condition.values[0]
    rep = str(metadata[metadata.filename == sample_name.split("/")[-1]].replicate.values[0])
    color_of_sample = color_palette[cond]

    noise_level = parameter.loc["noise_level", cond+"_"+rep]
    rescaling_factor = parameter.loc["rescaling_factor", cond+"_"+rep]

    data = MassSpecData()
    data.add_raw_spectrum(sample_name)
                
    masses = mass_shifts_df["masses "+cond+"_"+rep].dropna().values
    intensities = mass_shifts_df["raw intensities "+cond+"_"+rep].dropna().values

    x_gauss_func = np.arange(mass_range_start, mass_range_end)
    y_gauss_func = utils.multi_gaussian(x_gauss_func, intensities, masses, stddev_isotope_distribution)
        
    axes[order_in_plot[ix]].plot(data.raw_spectrum[:,0], flip_spectrum[ix]*data.raw_spectrum[:,1]/rescaling_factor, label=cond_mapping[cond], color=color_of_sample)
    axes[order_in_plot[ix]].plot(masses, flip_spectrum[ix]*intensities, '.', color='0.3')
    axes[order_in_plot[ix]].plot(x_gauss_func,flip_spectrum[ix]*y_gauss_func, color='0.3')
    axes[order_in_plot[ix]].axhline(y=flip_spectrum[ix]*noise_level, c='r', lw=0.2)
    if flip_spectrum[ix] > 0: axes[order_in_plot[ix]].legend(loc='upper right')

ylim_max = mass_shifts_df.filter(regex="raw intensities.*").max().max()      
plt.xlim((mass_range_start, mass_range_end))
plt.ylim((-ylim_max*1.18, ylim_max*1.18))
plt.xlabel("mass (Da)"); plt.ylabel("rel. intensity") 
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

fig = plt.figure(figsize=(7, 1.8))
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
