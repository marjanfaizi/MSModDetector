#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 5 2022

@author: Marjan Faizi
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from simulate_data import SimulateData
import utils

sns.set_color_codes("pastel")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1}                  
sns.set_context("paper", rc = paper_rc)   


aa_sequence_str = utils.read_fasta('../data/fasta_files/P04637.fasta')
modifications_table =  pd.read_csv('../data/modifications/modifications_P04637.csv', sep=';')
error_estimate_table = pd.read_csv("../output/noise_distribution_table.csv")

modform_distribution =  pd.read_csv('../data/modform_distributions/modform_distribution_phospho.csv', sep=',')

data_simulation = SimulateData(aa_sequence_str, modifications_table)

sigma_mean = error_estimate_table[error_estimate_table["sigma noise"]<0.4]["sigma noise"].mean()
sigma_std = error_estimate_table[error_estimate_table["sigma noise"]<0.4]["sigma noise"].std()
data_simulation.set_sigma_mean(sigma_mean)


data_simulation.reset_noise_levels()
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.add_noise(basal_noise_beta=10, vertical_noise_std=0.25, sigma_noise_std=sigma_std, horizontal_noise_std=0.1)
masses2, intensities2 = data_simulation.create_mass_spectrum(modform_distribution)

data_simulation.add_noise(basal_noise_beta=10, vertical_noise_std=0.125, sigma_noise_std=sigma_std/2, horizontal_noise_std=0.05)
masses3, intensities3 = data_simulation.create_mass_spectrum(modform_distribution)


plt.figure(figsize=(7,3))
plt.plot(masses3, intensities3, "g.-")
#plt.plot(masses2, intensities2, "r.-")
#plt.plot(masses, intensities, "b.-")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43630,44450))
plt.ylim(ymin=-0.005)
plt.tight_layout()
sns.despine()
plt.show()







data_simulation.save_mass_spectrum(masses, intensities, "../output/phospho_with_noise.csv")







