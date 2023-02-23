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
from pyopenms import AASequence
from scipy import stats


sys.path.append("../../src/")
from mass_spec_data import MassSpecData
import utils


### set figure style
sns.set()
sns.set_style("white")
sns.set_context("paper")


###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 2 ##############################################
###################################################################################################################
### isotopic distribution of unmodified p53
fasta_file_name = "P04637.fasta"
protein_entries = utils.read_fasta("../../fasta_files/"+fasta_file_name)
protein_sequence = list(protein_entries.values())[0]
mean_p53, stddev_p53 = utils.isotope_distribution_fit_par(protein_sequence, 100)
distribution = utils.get_theoretical_isotope_distribution(AASequence.fromString(protein_sequence), 100)
mass_grid, intensities = utils.fine_grain_isotope_distribution(distribution, 0.2, 0.02)

threshold = intensities.max()*(2/3)

plt.figure(figsize=(5.2,2.))
plt.plot(mass_grid, intensities, '-', color="0.3", label="isotopic distribution of p53")
plt.plot(mass_grid, utils.gaussian(mass_grid, intensities.max(), mean_p53, stddev_p53), 'r-', 
         label="Gaussian fit") 
plt.plot(mean_p53, intensities.max(), "r.")
plt.axhline(y=threshold, ls="--", color="0.3", lw=0.5)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43630, 43680))
plt.legend(frameon=False)
plt.tight_layout()
sns.despine()
plt.show()

### small protein example: >sp|Q9BZX4-2|ROP1B_HUMAN Isoform 2 of Ropporin-1B OS=Homo sapiens OX=9606 GN=ROPN1B
small_protein_sequence = "MWKVVNLPTDLFNSVMNVGRFTEEIEWLKFLALACSALGVTITKTLKIVCEVLSCDHNGGLPRIPFSTFQFLYTYIAEVDGEICASHVSRMLNYIEQEVIGPDGLITVNDFTQNPRVWLE"
mean_small_protein, stddev_small_protein = utils.isotope_distribution_fit_par(small_protein_sequence, 100)
distribution = utils.get_theoretical_isotope_distribution(AASequence.fromString(small_protein_sequence), 100)
mass_grid, intensities = utils.fine_grain_isotope_distribution(distribution, 0.2, 0.02)

threshold = intensities.max()*(2/3)

plt.figure(figsize=(5.2,2.))
plt.plot(mass_grid, intensities, '-', color="0.3", label="isotopic distribution of small protein")
plt.plot(mass_grid, utils.gaussian(mass_grid, intensities.max(), mean_small_protein, stddev_small_protein), 'r-', 
         label="Gaussian fit") 
plt.plot(mean_small_protein, intensities.max(), "r.")
plt.axhline(y=threshold, ls="--", color="0.3", lw=0.5)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((13577, 13610))
plt.legend(frameon=False)
plt.tight_layout()
sns.despine()
plt.show()

###################################################################################################################
###################################################################################################################






###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 3 ##############################################
###################################################################################################################
### mass spectrum example from p53 from contol condition with Nutlin only
sample_name = "../../raw_data/nutlin_only_rep5.mzml"
data = MassSpecData()
data.add_raw_spectrum(sample_name)

plt.figure(figsize=(7, 3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '-', color="0.3", linewidth=0.8)
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("intensity (a.u.)", fontsize=10)
#plt.xlim((2500,50000))
plt.ylim((-30, 2200))
plt.tight_layout()
sns.despine()
plt.show()

plt.figure(figsize=(4.5, 1.8))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '-', color="0.3", linewidth=0.8)
plt.xlabel("mass (Da)", fontsize=9)
plt.ylabel("intensity (a.u.)", fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlim((43650, 44550))
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
plt.xlim((44550, 46050))
plt.ylim((-5, 110))
plt.tight_layout()
sns.despine()
plt.show()
###################################################################################################################
###################################################################################################################



###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 4 ##############################################
###################################################################################################################
### noise and error distribution
error_estimate_table = pd.read_csv("../output/error_noise_distribution_table.csv")

lw = 1.3
binsize = 25

fig, axes = plt.subplots(1, 3, figsize=(7, 2))
# basal noise
basal_noise = error_estimate_table["basal_noise"].values
axes[0].hist(basal_noise, bins=binsize, density=True, alpha=0.5)
axes[0].set_xlabel("basal noise (rel.)")
axes[0].set_ylabel("density")
x = np.linspace(-1e-3, basal_noise.max(), len(basal_noise))
a, b, loc, scale = stats.beta.fit(basal_noise)  
pdf_beta = stats.beta.pdf(x, a, b, loc, scale)  
axes[0].plot(x, pdf_beta, color="purple", lw=lw)
# horizontal error
horizontal_error = error_estimate_table[(error_estimate_table["horizontal_error"]<0.3) &
                                        (error_estimate_table["horizontal_error"]>-0.3) &
                                        (error_estimate_table["is_signal"]==True)]["horizontal_error"].values
axes[1].hist(horizontal_error, bins=binsize, density=True, alpha=0.5)
axes[1].set_xlabel("horizontal error (Da)")
axes[1].set_ylabel("density")
x = np.linspace(-0.3, 0.3, len(horizontal_error))
a, b, loc, scale = stats.beta.fit(horizontal_error[(horizontal_error>-0.2) & (horizontal_error<0.2)])  
pdf_beta = stats.beta.pdf(x, a, b, loc, scale) 
axes[1].plot(x, pdf_beta, color="purple", lw=lw)
# vertical error
vertical_error = error_estimate_table[(error_estimate_table["vertical_error"]<0.25) &
                                        (error_estimate_table["vertical_error"]>-0.25) &
                                        (error_estimate_table["is_signal"]==True)]["vertical_error"].values
axes[2].hist(vertical_error, bins=binsize, density=True, alpha=0.5)
axes[2].set_xlabel("vertical error (rel.)")
axes[2].set_ylabel("density")
x = np.linspace(-0.25, 0.25, len(vertical_error))
a, b, loc, scale = stats.beta.fit(vertical_error)
pdf_beta = stats.beta.pdf(x, a, b, loc, scale)  
axes[2].plot(x, pdf_beta, color="purple", lw=lw)

sns.despine()
plt.tight_layout()
plt.show()
###################################################################################################################
###################################################################################################################



###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 5 ##############################################
###################################################################################################################
modform_file_name = "complex"
modform_distribution = pd.read_csv("../simulated_data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")

# "arr_0": mass_shift, "arr_1": chi_sqaure_score, "arr_2": mass_shift_deviation, "arr_3": ptm_patterns
# "arr_4": ptm_patterns_top3,  "arr_5": ptm_patterns_top5,  "arr_6": ptm_patterns_top10
#npzfile = np.load("../output/evaluated_complex_data_min_ptm.npz")
npzfile = np.load("../output/evaluated_complex_data_100_simulations_min_ptm_36ppm.npz")

repeats = npzfile["arr_0"].shape[0]

metric = pd.DataFrame({"PTM pattern": npzfile["arr_3"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

metric = pd.DataFrame({"PTM pattern": npzfile["arr_4"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped_top3 = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

metric = pd.DataFrame({"PTM pattern": npzfile["arr_5"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped_top5 = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

metric = pd.DataFrame({"PTM pattern": npzfile["arr_6"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped_top10 = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

xlabels = [int(item) for item in ptm_pattern_grouped.index.tolist()]

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 4.5))
ptm_pattern_grouped.plot.bar(stacked=True, ax=ax[0], legend=None)
ptm_pattern_grouped_top3.plot.bar(stacked=True, ax=ax[1], legend=None)
ptm_pattern_grouped_top5.plot.bar(stacked=True, ax=ax[2], legend=None)
ptm_pattern_grouped_top10.plot.bar(stacked=True, ax=ax[3], legend=None)
ax[3].set_xticklabels(xlabels, rotation=45)
ax[3].set_xlabel("mass shift [Da]")
[ax[i].set_ylabel("# simulations") for i in range(4)]
sns.despine()
fig.tight_layout()
plt.show()


# additional information 
amount_simulations = 100

amount_mass_shifts = ptm_pattern_grouped.shape[0]
print(ptm_pattern_grouped[0.75<1-(ptm_pattern_grouped["not detected"]/amount_simulations)].shape[0], 
      "mass shifts detected out of", amount_mass_shifts)

print("Results only for the best solution:")
print(ptm_pattern_grouped[0.75<(ptm_pattern_grouped["true positive"]/amount_simulations)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")

print("Results only for the top 3solution:")
print(ptm_pattern_grouped_top3[0.75<(ptm_pattern_grouped_top3["true positive"]/amount_simulations)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")

print("Results only for the top 5 solution:")
print(ptm_pattern_grouped_top5[0.75<(ptm_pattern_grouped_top5["true positive"]/amount_simulations)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")
print("Results only for the top 10 solution:")
print(ptm_pattern_grouped_top10[0.75<(ptm_pattern_grouped_top10["true positive"]/amount_simulations)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")
###################################################################################################################
###################################################################################################################



###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 6 ##############################################
###################################################################################################################
modform_file_name = "complex"
modform_distribution = pd.read_csv("../simulated_data/ptm_patterns/ptm_patterns_"+modform_file_name+".csv", sep=",")

# "arr_0": mass_shift, "arr_1": chi_sqaure_score, "arr_2": mass_shift_deviation, "arr_3": ptm_patterns
# "arr_4": ptm_patterns_top3,  "arr_5": ptm_patterns_top5,  "arr_6": ptm_patterns_top10
npzfile =  np.load("../output/evaluated_complex_data_100_simulations_min_both_36ppm.npz")

repeats = npzfile["arr_0"].shape[0]

metric = pd.DataFrame({"PTM pattern": npzfile["arr_3"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

metric = pd.DataFrame({"PTM pattern": npzfile["arr_4"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped_top3 = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

metric = pd.DataFrame({"PTM pattern": npzfile["arr_5"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped_top5 = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)

metric = pd.DataFrame({"PTM pattern": npzfile["arr_6"].flatten(), "mass_shift": modform_distribution.mass.tolist()*repeats})
metric["PTM pattern"].mask(metric["PTM pattern"] == 1.0, "true positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"] == 0.0, "false positive", inplace=True)
metric["PTM pattern"].mask(metric["PTM pattern"].isna(), "not detected", inplace=True)
ptm_pattern_grouped_top10 = metric.groupby(["mass_shift", "PTM pattern"]).size().unstack(fill_value=0)


xlabels = [int(item) for item in ptm_pattern_grouped.index.tolist()]

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 4.5))
ptm_pattern_grouped.plot.bar(stacked=True, ax=ax[0], legend=None)
ptm_pattern_grouped_top3.plot.bar(stacked=True, ax=ax[1], legend=None)
ptm_pattern_grouped_top5.plot.bar(stacked=True, ax=ax[2], legend=None)
ptm_pattern_grouped_top10.plot.bar(stacked=True, ax=ax[3], legend=None)
ax[3].set_xticklabels(xlabels, rotation=45)
ax[3].set_xlabel("mass shift [Da]")
[ax[i].set_ylabel("# simulations") for i in range(4)]
sns.despine()
fig.tight_layout()
plt.show()


# additional information 
amount_simulations = 100

amount_mass_shifts = ptm_pattern_grouped.shape[0]
print(ptm_pattern_grouped[0.75<1-(ptm_pattern_grouped["not detected"]/amount_simulations)].shape[0], 
      "mass shifts detected out of", amount_mass_shifts)

print("Results only for the best solution:")
print(ptm_pattern_grouped[0.75<(ptm_pattern_grouped["true positive"]/amount_simulations)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")

print("Results only for the top 3 solution:")
print(ptm_pattern_grouped_top3[0.75<(ptm_pattern_grouped_top3["true positive"]/amount_simulations)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")

print("Results only for the top 5 solution:")
print(ptm_pattern_grouped_top5[0.75<(ptm_pattern_grouped_top5["true positive"]/amount_simulations)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")
print("Results only for the top 10 solution:")
print(ptm_pattern_grouped_top10[0.75<(ptm_pattern_grouped_top10["true positive"]/amount_simulations)].shape[0], 
      "true positive PTM pattern predictions", amount_mass_shifts, "mass shifts")
###################################################################################################################
###################################################################################################################


###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 7 ##############################################
###################################################################################################################
"""
Mass shift (Da) & PTM pattern & Mass error (Da) & Bin size (Da) & ${\#}$Combinations\\
120 & 1[Cys] & 1.553 & 1.109 & 47 \\ 
136 & 1[Ox]1[Cys] & 1.063 & 0 & 35 \\ 
180 & 1[Ph]2[Me3]1[Ox] & 0.141 & 0.191 & 84 \\ 
200 & 1[Ph]1[Cys] & 1.569 & 2.234 & 262 \\ 
216 & 1[Ph-OH]1[Cys] & 0.400 & 1.324 & 294 \\ 
259 & 1[Ph-OH]1[Me3]1[Cys] & 0.095 & 0 & 685 \\ 
297 & 1[Ph]1[Ph-OH]1[Cys] & 0.711 & 0.323 & 949 \\ 
309 & 3[Ph-OH]1[Ox] & 0.043 & 0 & 1028 \\ 
357 & 3[Cys] & 0.012 & 0.668 & 2251\\ 
377 & 2[Ph]1[Ph-OH]1[Cys] & 0.179 & 1.351 & 2324 \\ 
394 & 1[Ph]2[Ph-OH]1[Cys] & 0.561 & 2.026 & 2585 \\
435 & 2[Ph-OH]2[Cys] & 1.413 & 0.351 & 4050 \\ 
454 & 1[Ph-OH]3[Cys] & 1.062 & 3.326 & 3411 \\ 
468 & 1[Ph-OH]1[Me1]3[Cys] & 0.824 & 0 & 4229 \\ 
473 & 1[Ph]4[Ph-OH] & 1.104 & 1.521 & 4636 \\ 
477 & 4[Cys] & 0.493 & 2.499 & 4853 \\ 
493 & 1[Ph]3[Ph-OH]1[Cys] & 0.759 & 0.691 & 4899 \\ 
516 & 2[Ph]3[Cys] & 0.650 & 0 & 5073 \\ 
534 & 1[Ph]1[Ph-OH]3[Cys] & 0.550 & 1.555 & 5668 \\ 
553 & 2[Ph-OH]3[Cys] & 0.259 & 3.011 & 5958 \\ 
569 & 2[Ph-OH]1[Ox]3[Cys] & 0.106 & 0 & 6354 \\ 
610 & 1[Me1]5[Cys] & 0.700 & 0 & 6842 \\ 
613 & 1[Ph]3[Ph-OH]2[Cys] & 1.536 & 0.221 & 6846 \\ 
633 & 1[Ph]2[Ph-OH]3[Cys] & 0.328 & 1.865 & 7622 \\ 
652 & 3[Ph-OH]3[Cys] & 1.054 & 1.191 & 7714 \\ 
692 & 1[Ph-OH]5[Cys] & 1.231 & 0.170 & 8226 \\ 
723 & 1[Ph-OH]1[Me1]1[Ox]5[Cys] & 0.076 & 0 & 9008 \\
731 & 1[Ph]3[Ph-OH]3[Cys] & 0.359 & 0 & 9309 \\ 
792 & 2[Ph-OH]5[Cys] & 1.186 & 0 & 10166 \\ 
812 & 3[Ph-OH]1[Ac]4[Cys] & 0.074 & 0 & 10499 \\ 
"""

mass_shifts_df = pd.read_csv("../../output/mass_shifts_min_ptm.csv", sep=",")

x_values = mass_shifts_df["mass shift"].tolist()
y_values = [47, 35, 84, 262, 294, 685, 949, 1028, 2251, 2324, 2585, 4050, 3411, 4229, 4636, 4853, 
            4899, 5073, 5668, 5958, 6354, 6842, 6846, 7622, 7714, 8226, 9008, 9309, 10166, 10499]

plt.figure(figsize=(5.2,2.))
plt.plot(x_values, y_values, 'o', color="0.3")
plt.xlabel("mass shift (Da)")
plt.ylabel("${\#}$ combinations")
plt.tight_layout()
sns.despine()
plt.show()
###################################################################################################################
###################################################################################################################




