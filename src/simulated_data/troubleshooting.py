#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 2022

@author: Marjan Faizi
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
sys.path.append("../../")

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
import utils
from simulate_data import SimulateData
import config 


modform_file_name = "complex"


### PTM database
modifications_table = pd.read_csv("../../"+config.modfication_file_name, sep=";")
modifications_table["unimod_id"] = modifications_table["unimod_id"].astype("Int64")

### simulated PTM patterns
modform_distribution = pd.read_csv("ptm_patterns/ptm_patterns_"+modform_file_name+".csv", 
                                   sep=",")
modform_distribution["rel. intensity"] = modform_distribution["intensity"]/ modform_distribution["intensity"].sum()

### estimated error and noise 
error_estimate_table = pd.read_csv("../../output/error_noise_distribution_table.csv")
basal_noise = error_estimate_table["basal_noise"].values
horizontal_error = error_estimate_table[(error_estimate_table["horizontal_error"]<0.3) &
                                        (error_estimate_table["horizontal_error"]>-0.3) &
                                        (error_estimate_table["is_signal"]==True)]["horizontal_error"].values
vertical_error = error_estimate_table[(error_estimate_table["is_signal"]==True)]["vertical_error"].values
vertical_error_par = list(stats.beta.fit(vertical_error))
horizontal_error_par = list(stats.beta.fit(horizontal_error)) 
basal_noise_par = list(stats.beta.fit(basal_noise))    



protein_entries = utils.read_fasta("../../"+config.fasta_file_name)
protein_sequence = list(protein_entries.values())[0]
unmodified_species_mass, stddev_isotope_distribution = utils.isotope_distribution_fit_par(protein_sequence, 100)
mass_tolerance = config.mass_error_ppm*1e-6*unmodified_species_mass
    
mod = Modifications("../../"+config.modfication_file_name, protein_sequence)

data_simulation = SimulateData(protein_sequence, modifications_table)

data_simulation.reset_noise_error()
data_simulation.add_noise(vertical_error_par=vertical_error_par, horizontal_error_par=horizontal_error_par,
                          basal_noise_par=basal_noise_par)

masses, intensities = data_simulation.create_mass_spectrum(modform_distribution)
spectrum = np.column_stack((masses, intensities))

data = MassSpecData(spectrum)
data.set_mass_range_of_interest(data.raw_spectrum[0,0], data.raw_spectrum[-1,0])
all_peaks = data.picking_peaks()
peaks_normalized = data.preprocess_peaks(all_peaks, config.distance_threshold_adjacent_peaks)
noise_level = config.noise_level_fraction*peaks_normalized[:,1].std()
peaks_normalized_above_noise = peaks_normalized[peaks_normalized[:,1]>noise_level]

gaussian_model = GaussianModel("simulated", stddev_isotope_distribution)
gaussian_model.determine_variable_window_sizes(unmodified_species_mass, config.window_size_lb, config.window_size_ub)
gaussian_model.fit_gaussian_within_window(peaks_normalized_above_noise, config.pvalue_threshold)      
gaussian_model.refit_results(peaks_normalized, noise_level, refit_mean=True)
gaussian_model.calculate_relative_abundaces(data.search_window_start, data.search_window_end)

mass_shifts = MassShifts(data.raw_spectrum[0,0], data.raw_spectrum[-1,0])
mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results,  "simulated")

mass_shifts.calculate_avg_mass()
mass_shifts.add_mass_shifts(unmodified_species_mass)
mass_shifts.determine_ptm_patterns(mod, mass_tolerance, config.objective_fun, config.laps_run_lp)     
mass_shifts.add_ptm_patterns_to_table()

fig = plt.figure(figsize=(7, 2.5))
plt.plot(data.raw_spectrum[:, 0], data.raw_spectrum[:, 1], '-', color="0.3")
plt.plot(modform_distribution["mass"]+unmodified_species_mass, modform_distribution["intensity"], '.', markersize=3, color="r")
plt.plot(gaussian_model.fitting_results["mean"], gaussian_model.fitting_results["amplitude"]*data.rescaling_factor, '.', color="b")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43670, 44450))
sns.despine()
fig.tight_layout()
plt.show()
