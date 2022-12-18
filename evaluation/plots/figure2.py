#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 2022

@author: Marjan Faizi
"""

import sys
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../")
sys.path.append("../simulated_data")

from simulate_data import SimulateData
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


### figure B
modform_distribution = pd.read_csv("../simulated_data/ptm_patterns/ptm_patterns_toy_example.csv", sep=",")

modform_distribution["rel. intensity"] = modform_distribution["intensity"]/modform_distribution["intensity"].sum()

data_simulation = SimulateData(protein_sequence, modifications_table)
data_simulation.reset_noise_error()
data_simulation.add_noise(horizontal_error_par=horizontal_error_par, basal_noise_par=basal_noise_par, 
                          vertical_error_par=vertical_error_par)
masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)

xrange = [ np.arange(i-20, i+20) for i in modform_distribution["mass"].values+unmodified_species_mass ]
y = [ utils.gaussian(xrange[i], modform_distribution["rel. intensity"].values[i],  
                     modform_distribution["mass"].values[i]+unmodified_species_mass, stddev_isotope_distribution) 
      for i in range(len(modform_distribution)) ]


fig = plt.figure(figsize=(2.4, 2.2))
plt.plot(masses, intensities/modform_distribution["intensity"].sum(), color="0.3") 
#plt.plot(np.transpose(xrange), np.transpose(np.array(y)), "-r", linewidth=.6) 
plt.vlines(x=modform_distribution["mass"]+unmodified_species_mass, ymin=0, ymax=0.7, color="0.1", ls="--", lw=0.5)
#plt.plot(modform_distribution["mass"]+unmodified_species_mass, modform_distribution["rel. intensity"], ".r", markersize=4) 
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim([43630, 43850])
plt.ylim([-0.01, 0.7])
fig.tight_layout()
sns.despine()
plt.show()


