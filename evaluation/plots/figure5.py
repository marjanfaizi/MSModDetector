#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 2022

@author: Marjan Faizi
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append("../../src")
sys.path.append("../simulated_data")

from simulate_data import SimulateData
from modifications import Modifications
import utils


### set figure style
sns.set()
sns.set_style("white")
sns.set_context("paper")


### required input
fasta_file_name = "P04637.fasta"
modfication_file_name = "modifications_P04637.csv"

error_estimate_table = pd.read_csv("../output/error_noise_distribution_table.csv")
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

modifications_table = pd.read_csv("../../modifications/"+modfication_file_name, sep=";")
modifications_table["unimod_id"] = modifications_table["unimod_id"].astype("Int64")

protein_entries = utils.read_fasta("../../fasta_files/"+fasta_file_name)
protein_sequence = list(protein_entries.values())[0]    
mod = Modifications("../../modifications/"+modfication_file_name, protein_sequence)

unmodified_species_mass, stddev_isotope_distribution = utils.isotope_distribution_fit_par(protein_sequence, 100)


### figure A: simulated spectrum of complex PTM patterns
modform_file_name = "complex"
modform_distribution = pd.read_csv("../simulated_data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")

data_simulation = SimulateData(protein_sequence, modifications_table)
data_simulation.reset_noise_error()
data_simulation.add_noise(horizontal_error_par=horizontal_error_par, basal_noise_par=basal_noise_par, 
                          vertical_error_par=vertical_error_par)
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

plt.figure(figsize=(7, 1.8))
plt.plot(masses, intensities, color="0.3") 
plt.plot(modform_distribution["mass"]+unmodified_species_mass, modform_distribution["intensity"], ".r", markersize=2) 
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43740,44355))
plt.ylim([-10, 990])
plt.tight_layout()
sns.despine()
plt.show()


### figure B: best PTM pattern solution and 5 best PTM pattern solutions
# "arr_0": mass_shift, "arr_1": chi_sqaure_score, "arr_2": mass_shift_deviation, "arr_3": ptm_patterns
# "arr_4": ptm_patterns_top3,  "arr_5": ptm_patterns_top5,  "arr_6": ptm_patterns_top10
npzfile = np.load("../output/evaluated_complex_data_50_simulations.npz")

repeats = npzfile["arr_0"].shape[0]

metric = pd.DataFrame({"PTM pattern": npzfile["arr_3"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

metric = pd.DataFrame({"PTM pattern": npzfile["arr_5"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped_top5 = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 2.8))
ptm_pattern_grouped.plot.bar(stacked=True, ax=ax[0], legend=None)
ptm_pattern_grouped_top5.plot.bar(stacked=True, ax=ax[1], legend=None)
ax[1].set_xticklabels(np.arange(1,len(modform_distribution)+1), rotation=45)
ax[1].set_xlabel("mass shift [Da]")
ax[0].set_ylabel("# simulations")
ax[1].set_ylabel("# simulations")
sns.despine()
fig.tight_layout()
plt.show()


# additional information 
amount_mass_shifts = ptm_pattern_grouped.shape[0]
print(ptm_pattern_grouped[0.5<1-(ptm_pattern_grouped["not detected"]/50)].shape[0], 
      "mass shifts detected out of", amount_mass_shifts)

print("Results only for the best solution:")
print(ptm_pattern_grouped[0.5<(ptm_pattern_grouped["true positive"]/50)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")

print("Results only for the top 5 solution:")
print(ptm_pattern_grouped_top5[0.5<(ptm_pattern_grouped_top5["true positive"]/50)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")



