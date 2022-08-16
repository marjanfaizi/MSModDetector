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


sys.path.append("..")
sys.path.append("../../")

from mass_spec_data import MassSpecData
import utils
import config

### set figure style
sns.set()
sns.set_style("white")
sns.set_context("paper")


###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 1 ##############################################
###################################################################################################################
### isotopic distribution of unmodified p53
protein_entries = utils.read_fasta("../../"+config.fasta_file_name)
protein_sequence = list(protein_entries.values())[0]
mean_p53, stddev_p53 = utils.isotope_distribution_fit_par(protein_sequence, 100)
distribution = utils.get_theoretical_isotope_distribution(AASequence.fromString(protein_sequence), 100)
mass_grid, intensities = utils.fine_grain_isotope_distribution(distribution, 0.2, 0.02)


plt.figure(figsize=(5.5,2.5))
plt.plot(mass_grid, intensities, '-', color="0.3", label="isotopic distribution of p53")
plt.plot(mass_grid, utils.gaussian(mass_grid, intensities.max(), mean_p53, stddev_p53), 'r-', 
         label="Gaussian fit") 
plt.plot(mean_p53, intensities.max(), "r.")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43630, 43680))
plt.legend(frameon=False)
plt.tight_layout()
sns.despine()
plt.show()
###################################################################################################################
###################################################################################################################



###################################################################################################################
############################################## SUPPLEMENTAL FIGURE 2 ##############################################
###################################################################################################################
### mass spectrum example from p53 from contol condition with Nutlin only
sample_name = "../../raw_data/nutlin_only_rep5.mzml"
data = MassSpecData()
data.add_raw_spectrum(sample_name)

plt.figure(figsize=(7, 3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '-', color="0.3", linewidth=0.8)
plt.xlabel("mass (Da)", fontsize=10)
plt.ylabel("intensity (a.u.)", fontsize=10)
plt.xlim((2500, 80000))
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
############################################## SUPPLEMENTAL FIGURE 3 ##############################################
###################################################################################################################
### noise and error distribution
error_estimate_table = pd.read_csv("../../output/error_noise_distribution_table.csv")

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


