# -*- coding: utf-8 -*-
"""
Created on Mar 30 2022

@author: Marjan Faizi
"""

import re
from itertools import product
import numpy as np
from scipy import optimize

import utils 
from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel


class ErrorEstimation(object):
    """
    This class estimates the following error and noise levels from given experimental data:
        - basal noise from a mass range wihtout any signal 
        - width, called sigma, of the single peaks within the mass range of interest
        - vertical error is estimated by the distance between the experimental peaks and the fitted gaussian curve
        - horizontal error is calculated as the distance between one peak to another 
    """

    def __init__(self, file_names, conditions, replicates):
        self.file_names = file_names
        self.conditions = conditions
        self.replicates = replicates
        self.sample_names= [cond+"_"+rep for cond, rep in product(self.conditions, self.replicates)]
        self.mass_start_range = 43600
        self.mass_end_range = 44600
        self.noise_start_range = 47000
        self.noise_end_range = 49000
        
        self.basal_noise = []
        self.sigma_sinlge_peak = []
        self.vertical_error = []
        self.horizontal_error = []

        
    def set_data_preprocessing_parameter(self, noise_level_fraction, distance_threshold_adjacent_peaks):
        self.distance_threshold_adjacent_peaks = distance_threshold_adjacent_peaks
        self.noise_level_fraction = noise_level_fraction


    def set_gaussian_model_parameter(self, stddev_isotope_distribution, unmodified_species_mass, pvalue_threshold):
        self.stddev_isotope_distribution = stddev_isotope_distribution
        self.unmodified_species_mass = unmodified_species_mass
        self.pvalue_threshold = pvalue_threshold
    

    def estimate_error(self):
        for rep in self.replicates:
            
            file_names_same_replicate = [file for file in self.file_names if re.search(rep, file)]
        
            for cond in self.conditions:
            
                sample_name = [file for file in file_names_same_replicate if re.search(cond, file)]        
                if sample_name:
                    sample_name = sample_name[0]
           
                    data = MassSpecData(sample_name)
                    data.set_search_window_mass_range(self.mass_start_range, self.mass_end_range)
                    data_in_search_window = data.determine_search_window(data.raw_spectrum)
                    data_in_search_window[:,1] = data_in_search_window[:,1] / data_in_search_window[:,1].max()
                    all_peaks = data.picking_peaks()
                    trimmed_peaks = data.remove_adjacent_peaks(all_peaks, self.distance_threshold_adjacent_peaks)
                    trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
                    trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])
                    
                    noise_level = self.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
                    gaussian_model = GaussianModel(cond, self.stddev_isotope_distribution)
                    gaussian_model.determine_adaptive_window_sizes(self.unmodified_species_mass)
                    gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, self.pvalue_threshold)
                    gaussian_model.remove_overlapping_fitting_results()
                    gaussian_model.refit_results(trimmed_peaks_in_search_window, refit_mean=True)
                    
        
                    noise_start_ix = data.find_nearest_mass_idx(data.raw_spectrum, self.noise_start_range)
                    noise_end_ix = data.find_nearest_mass_idx(data.raw_spectrum, self.noise_end_range)
                    self.basal_noise += (data.raw_spectrum[noise_start_ix:noise_end_ix,1]/data.rescaling_factor).tolist()
        
                    for peak in trimmed_peaks_in_search_window:
                        window_size = 0.6
                        selected_region_ix = np.argwhere((data_in_search_window[:,0]<=peak[0]+window_size) & (data_in_search_window[:,0]>=peak[1]-window_size))[:,0]
                        selected_region = data_in_search_window[selected_region_ix]
                        guess = 0.2
                        optimized_param, _ = optimize.curve_fit(lambda x, sigma: utils.gaussian(x, peak[1], peak[0], sigma), 
                                                                selected_region[:,0], selected_region[:,1], maxfev=100000,
                                                                p0=guess, bounds=(0, window_size))
                        self.sigma_sinlge_peak.append(optimized_param[0])
        
                    self.horizontal_error += (trimmed_peaks_in_search_window[1:,0]-trimmed_peaks_in_search_window[:-1,0]).tolist()
        
        
                    x_gauss_func = trimmed_peaks_in_search_window[:,0]
                    y_gauss_func = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results["amplitude"], gaussian_model.fitting_results["mean"], gaussian_model.stddev)
                    self.vertical_error += (y_gauss_func - trimmed_peaks_in_search_window[:,1]).tolist()
        
        self.basal_noise = np.sort(self.basal_noise)
        self.sigma_sinlge_peak = np.sort(self.sigma_sinlge_peak)
        self.horizontal_error = np.sort(self.horizontal_error)
        self.vertical_error = np.sort(self.vertical_error)


"""

from matplotlib import pyplot as plt

beta = 200
plt.hist(basal_noise, bins=2000)
plt.plot(basal_noise, beta*np.exp(-beta*basal_noise), "g-")
gfg = np.random.exponential(1/beta, 1000000)
plt.hist(gfg, 1000, color='r', density=True)

plt.hist(sigma_sinlge_peak, bins=500)
plt.plot(sigma_sinlge_peak, utils.gaussian(sigma_sinlge_peak, 6, sigma_sinlge_peak.mean(), sigma_sinlge_peak.std()), "r.-") 
plt.show() 

plt.hist(horizontal_error, bins=40)
plt.plot(horizontal_error, utils.gaussian(horizontal_error, 50, horizontal_error.mean(), horizontal_error.std()), "r.-") 

plt.hist(trimmed_peaks_in_search_window[:,1], bins=100)
plt.hist(vertical_error, bins=150)
plt.plot(vertical_error, utils.gaussian(vertical_error, 15, vertical_error.mean(), vertical_error.std()), "r.-") 

plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1]/data.rescaling_factor, '.-', color="0.3", linewidth=1)
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((46000,50000))
plt.ylim(ymin=-0.005)
plt.tight_layout()
plt.show()
"""





