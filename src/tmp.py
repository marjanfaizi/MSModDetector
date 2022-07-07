#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2021

@author: Marjan Faizi
"""

import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
import utils
import config as config
from scipy import optimize

#modform_distribution =  pd.read_csv('../data/modifications/modform_distribution.csv', sep=',')
#mod_amplitude = modform_distribution["intensity"]
#mod_mean = modform_distribution["mass"]+config.unmodified_species_mass

file_name = [file for file in glob.glob(config.file_names)][1] 
rep = config.replicates[0]
cond = config.conditions[0]

protein_entries = utils.read_fasta(config.fasta_file_name)
protein_sequence = list(protein_entries.values())[0]
mod = Modifications(config.modfication_file_name, protein_sequence)

mass_shifts = MassShifts(config.mass_start_range, config.mass_end_range)
data = MassSpecData(file_name)
data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)        

all_peaks = data.picking_peaks()
trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])

noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
trimmed_peaks_in_search_window_above_noise = trimmed_peaks_in_search_window[trimmed_peaks_in_search_window[:,1]>noise_level]

gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
gaussian_model.determine_variable_window_sizes(config.unmodified_species_mass, config.window_size_lb, config.window_size_ub)
gaussian_model.fit_gaussian_within_window(trimmed_peaks_in_search_window_above_noise, config.allowed_overlap_fitting_window, config.pvalue_threshold, noise_level)      
gaussian_model.refit_results(trimmed_peaks_in_search_window_above_noise, noise_level, refit_mean=True)



all_points = data.raw_spectrum[(data.raw_spectrum[:,0]>=config.mass_start_range) & (data.raw_spectrum[:,0]<=config.mass_end_range)][:,0]

mass_peak = all_peaks[(all_peaks[:,0]>=config.mass_start_range) & (all_peaks[:,0]<=config.mass_end_range)][:,0]
mean = gaussian_model.fitting_results["mean"].to_numpy()

diff_all_points = np.subtract.outer(all_points,all_points)[np.tril_indices(all_points.shape[0], k = -1)]
diff_mass_peaks = np.subtract.outer(mass_peak,mass_peak)[np.tril_indices(mass_peak.shape[0], k = -1)]
diff_mean = np.subtract.outer(mean,mean)[np.tril_indices(mean.shape[0], k = -1)]



plt.hist(diff_mean, bins=600)
p = []

for peak in trimmed_peaks_in_search_window:
    window_size = 0.5
    selected_region_ix = np.argwhere((data.masses<=peak[0]+window_size) & (data.masses>=peak[0]-window_size))[:,0]
    masses = data.masses[selected_region_ix]; intensities = data.intensities[selected_region_ix]/data.rescaling_factor
    guess = 0.25
    optimized_param, _ = optimize.curve_fit(lambda x, sigma: utils.gaussian(x, peak[1], peak[0], sigma), 
                                            masses, intensities, maxfev=1000000,
                                            p0=guess, bounds=(0, window_size))
    p.append(optimized_param[0])

plt.hist(p,bins=20)



mean = gaussian_model.fitting_results["mean"]
amplitude = gaussian_model.fitting_results["amplitude"]

gaussian_model.remove_overlapping_fitting_results()

mean2 = gaussian_model.fitting_results["mean"]
amplitude2 = gaussian_model.fitting_results["amplitude"]


gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)

####todo refit mean and compare with actual modform masses   
error_func_mean = lambda mean, x, y: (utils.multi_gaussian(x, amplitude*data.rescaling_factor, mean, config.stddev_isotope_distribution) - y)**2
refitted_mean = optimize.least_squares(error_func_mean, bounds=(mean-5, mean+5), x0=mean, 
                                       args=(all_peaks[:,0], all_peaks[:,1]))
error_func_amp = lambda amplitude, x, y: (utils.multi_gaussian(x, amplitude, mean, config.stddev_isotope_distribution) - y)**2
refitted_amp = optimize.least_squares(error_func_amp, bounds=(0, 1), x0=amplitude*data.rescaling_factor, 
                                       args=(all_peaks[:,0], all_peaks[:,1]))
"""
mass_shifts.add_identified_masses_to_df(gaussian_model.fitting_results, data.rescaling_factor, cond+"_"+rep)


mass_shifts.calculate_avg_mass()

if config.bin_peaks == True:
    mass_shifts.bin_peaks()

if config.calculate_mass_shifts == True:
    mass_shifts.add_mass_shifts()

if config.determine_ptm_patterns == True:
    print("\nSearching for PTM combinations:")

    mass_shifts.determine_ptm_patterns(mod, config.mass_tolerance, config.objective_fun)        
    mass_shifts.add_ptm_patterns_to_table()
    mass_shifts.save_table(mass_shifts.ptm_patterns_df, "../output/ptm_patterns_table.csv")
    
mass_shifts.save_table(mass_shifts.identified_masses_df, "../output/mass_shifts.csv")
    
"""

plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '-', color="0.3", linewidth=1)
#plt.plot(mod_mean, mod_amplitude/200*data.rescaling_factor, 'b.')
#plt.plot(mean, amplitude*data.rescaling_factor, 'b.')
#plt.plot(mean2, amplitude2*data.rescaling_factor, 'r.')
#plt.plot(gaussian_model.fitting_results["mean"],gaussian_model.fitting_results["amplitude"]*data.rescaling_factor, 'g.')
#plt.plot(data.raw_spectrum[:,0], utils.multi_gaussian(data.raw_spectrum[:,0], refitted_amp.x, refitted_mean.x, config.stddev_isotope_distribution), 'g-')
#plt.plot(refitted_mean.x, refitted_amp.x, 'g.')
#plt.axhline(y=noise_level*data.rescaling_factor, c='r', lw=0.3)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43600,44540))
plt.tight_layout()
plt.show()

#####################################################################################################


modform_distribution =  pd.read_csv('../data/modifications/modform_distribution.csv', sep=',')
simulated_data =  pd.read_csv('../data/simulated_data_rep1.csv', sep=',', header=None)

mass_start_range = 43600
mass_end_range = 44540

data = MassSpecData('../data/simulated_data_rep1.csv')
data.set_search_window_mass_range(mass_start_range, mass_end_range)        

all_peaks = data.picking_peaks()
trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])


amplitudes = modform_distribution["intensity"]
means = modform_distribution["mass"]+config.unmodified_species_mass

x = np.arange(mass_start_range, mass_end_range)
y = utils.multi_gaussian(x, modform_distribution["intensity"], means, config.stddev_isotope_distribution)


plt.figure(figsize=(7,3))
#plt.plot(simulated_data[0], simulated_data[1], '.-', color="0.3", linewidth=1)

plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color="0.3", linewidth=1)
plt.plot(x, y/950, 'r-')
plt.plot(means, amplitudes/950, 'r.')
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((mass_start_range, mass_end_range))
plt.tight_layout()
plt.show()


##################################

def multi_gaussian(x, mean, stddev, *amplitude):
    y = np.zeros(len(x))
    for i in range(len(mean)):
        y += amplitude[i] * np.exp(-0.5*((x - mean[i]) / stddev)**2)

    return y  

def multi_gaussian3(x, stddev, amplitude, *mean):
    y = np.zeros(len(x))
    for i in range(len(mean)):
        y += amplitude[i] * np.exp(-0.5*((x - mean[i]) / stddev)**2)

    return y  


def multi_gaussian2(x, stddev, *params):
    y = np.zeros(len(x))
    for i in range(0, len(params), 2):
        amplitude = params[i]
        mean = params[i+1]
        y += amplitude * np.exp(-0.5*((x - mean) / stddev)**2)

    return y  



mass_exp = trimmed_peaks_in_search_window[:,0]
intensity_exp = trimmed_peaks_in_search_window[:,1]

"""
# CURVE FITTING WITH ALL PEAKS
popt, _ = optimize.curve_fit(lambda x, *amplitude: multi_gaussian(x, mass_exp, config.stddev_isotope_distribution,*amplitude), 
                             mass_exp, intensity_exp, p0=intensity_exp, bounds=(0, 1), maxfev=100000)
fit = multi_gaussian(mass_exp, config.stddev_isotope_distribution, *popt)

init = all_peaks.flatten()
popt2, _ = optimize.curve_fit(lambda x, *params: multi_gaussian2(x, config.stddev_isotope_distribution, *params), 
                             mass_exp, intensity_exp, p0=init, bounds=(0,), maxfev=100000)
fit2 = multi_gaussian(mass_exp, config.stddev_isotope_distribution, *popt2)

# CURVE FITTING WITH ONLY SELECTED PEAKS (ADJUSTING OF RESULTS) 
noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)
gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, config.pvalue_threshold)      
gaussian_model.remove_overlapping_fitting_results()

mean = gaussian_model.fitting_results["mean"]
amplitude = gaussian_model.fitting_results["amplitude"]

popt_s, _ = optimize.curve_fit(lambda x, *amplitude: multi_gaussian(x, mean, config.stddev_isotope_distribution,*amplitude), 
                             mass_exp, intensity_exp, p0=amplitude, bounds=(0, 1), maxfev=100000)
fit_s = multi_gaussian(mass_exp, mean, config.stddev_isotope_distribution, *popt_s)

init_s = np.concatenate((mean,amplitude))
popt2_s, _ = optimize.curve_fit(lambda x, *params: multi_gaussian2(x, config.stddev_isotope_distribution, *params), 
                             mass_exp, intensity_exp, p0=init_s, bounds=(0,np.inf), maxfev=100000)
fit2_s = multi_gaussian2(mass_exp, config.stddev_isotope_distribution, *popt2_s)


# LEASTSQUARES METHOD WITH ALL PEAKS
error_func = lambda par, x, y: (multi_gaussian(x, mass_exp, config.stddev_isotope_distribution, *par) - y)**2
popt, _ = optimize.leastsq(error_func, intensity_exp, bounds=(0,np.inf), args=(mass_exp, intensity_exp))
fit = multi_gaussian(mass_exp, config.stddev_isotope_distribution, *popt)

# LEASTSQUARES METHOD WITH SELECTED PEAKS ONLY
error_func_s = lambda par, x, y: (multi_gaussian(x, mean, config.stddev_isotope_distribution, *par) - y)**2
error_func2 = lambda par, x, y: (multi_gaussian2(x, config.stddev_isotope_distribution, *par) - y)**2

noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)
gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, config.pvalue_threshold)      
gaussian_model.remove_overlapping_fitting_results()

mean = gaussian_model.fitting_results["mean"]
amplitude = gaussian_model.fitting_results["amplitude"]

popt_s, _ = optimize.leastsq(error_func_s, x0=amplitude, args=(mass_exp, intensity_exp))
fit_s = multi_gaussian(mass_exp, mean, config.stddev_isotope_distribution, *popt_s)

init_s = np.concatenate((mean,amplitude))
popt2_s, _ = optimize.leastsq(error_func2, init_s, args=(mass_exp, intensity_exp))
fit2_s = multi_gaussian2(mean, config.stddev_isotope_distribution, *popt2_s)


plt.figure(figsize=(7,3))
#plt.plot(simulated_data[0], simulated_data[1], '.-', color="0.3", linewidth=1)
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color="0.3", linewidth=1)
plt.plot(x, y/950, 'r-')
plt.plot(means, amplitudes/950, 'r.')
plt.plot(popt_s, amplitude/5, 'b.')
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((mass_start_range, mass_end_range))
plt.tight_layout()
plt.show()

"""





