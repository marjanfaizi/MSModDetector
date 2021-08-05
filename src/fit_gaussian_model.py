#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 2 2021

@author: Marjan Faizi
"""

import numpy as np
from scipy import optimize
from scipy.stats import chisquare
from sklearn.metrics import r2_score

  
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
    
    
def multi_gaussian(x, amplitude, mean, stddev):
    y = np.zeros(len(x))
    for i in range(len(mean)):
        y += gaussian(x, amplitude[i], mean[i], stddev)

    return y  



class FitGaussianModel(object): 
    
    def __init__(self):
        self.stddev = 1.0
        self.maxfev = 100000
        self.chi_square_dof = 1
        self.sample_size_threshold = 5


    def set_standard_deviation(self, stddev):
        self.stddev = stddev


    def fit_gaussian(self, peaks, amplitude=None, mean=None, stddev=None):
        masses = peaks[:, 0]
        intensities = peaks[:, 1]
        
        index_most_abundant_peak = intensities.argmax()
        initial_amplitude = intensities[index_most_abundant_peak]
        intial_stddev = self.stddev
        initial_mean = masses[index_most_abundant_peak]
        
        if amplitude==None and mean==None and stddev==None:
            optimized_param, _ = optimize.curve_fit(gaussian, masses, intensities, maxfev=self.maxfev,
                                                    p0=[initial_amplitude, initial_mean, intial_stddev])
            
        elif amplitude==None and mean==None and stddev!=None:
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude, mean: gaussian(x, amplitude, mean, stddev), 
                                                    masses, intensities, maxfev=self.maxfev,
                                                    p0=[initial_amplitude, initial_mean])

        elif amplitude==None and mean!=None and stddev!=None:
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude: gaussian(x, amplitude, mean, stddev), 
                                                    masses, intensities, maxfev=self.maxfev,
                                                    p0=[initial_amplitude])
        
        return optimized_param


    def fit_gaussian_to_single_peaks(self, all_peaks, peaks_above_sn, fitting_window_sizes):
        masses = peaks_above_sn[:,0]
        fitting_results = {}
        for mass in masses:
            best_pvalue = -1.0
            for window_size in fitting_window_sizes:
                selected_region = self.select_region_to_fit(all_peaks, mass, window_size)
                sample_size = selected_region.shape[0]          
                if sample_size >= self.sample_size_threshold:
                    fitted_amplitude = self.fit_gaussian(selected_region, mean=mass, stddev=self.stddev)[0]               
                    pvalue = self.chi_square_test(selected_region, fitted_amplitude, mass)
                else:
                    pvalue = 0.0
                    fitted_amplitude = 0.0
    
                if pvalue > best_pvalue:
                    best_pvalue = pvalue
                    best_fitted_amplitude = fitted_amplitude
                    best_window_size = window_size
                          
            fitting_results[mass] = [best_fitted_amplitude, best_pvalue, best_window_size]
        return fitting_results    
     

    def select_region_to_fit(self, peaks, mass, fitting_window_size):
        all_masses = peaks[:,0]
        selected_region_ix = np.argwhere((all_masses<=mass+fitting_window_size) & (all_masses>=mass-fitting_window_size))[:,0]
        selected_region = peaks[selected_region_ix]
        return selected_region
        

    def chi_square_test(self, selected_region, amplitude, mean):
        observed_masses = selected_region[:,0]
        observed_intensities = selected_region[:,1]
        predicted_intensities = gaussian(observed_masses, amplitude, mean, self.stddev)
        chi_square_score = chisquare(observed_intensities, f_exp=predicted_intensities, ddof=self.chi_square_dof)
        return chi_square_score.pvalue
    
     
    def filter_fitting_results(self, fitting_results, pvalue_threshold):
        fitting_results_best_pvalues = self.select_best_pvalues(fitting_results, pvalue_threshold)
        if len(fitting_results_best_pvalues) > 1:
            fitting_results_reduced = self.remove_overlapping_fitting_results(fitting_results_best_pvalues)
            return fitting_results_reduced
        else:
            return fitting_results_best_pvalues
     
    def select_best_pvalues(self, fitting_results, pvalue_threshold):
        fitting_results_reduced = dict(filter(lambda elem: elem[1][1] >= pvalue_threshold, fitting_results.items()))
        return fitting_results_reduced
     
   
    def remove_overlapping_fitting_results(self, fitting_results):
        fitting_results_reduced = {}
        best_pvalue = 0.0 
        best_mean = list(fitting_results.keys())[0]   
        last_mean = list(fitting_results.keys())[-1]   
        all_window_sizes = np.append(0, np.array(list(fitting_results.values()))[:,2])
        ix_window_size = 0
        
        for current_mean, value in fitting_results.items():
            pvalue = value[1]
            #window_size = value[2]
            window_size = max(all_window_sizes[ix_window_size], all_window_sizes[ix_window_size+1])
            ix_window_size += 1
            if np.abs(best_mean-current_mean) <= window_size and pvalue > best_pvalue:
                best_mean = current_mean
                best_amplitude = value[0]
                best_pvalue = pvalue
        
            elif np.abs(best_mean-current_mean) > window_size or best_mean == last_mean:
                fitting_results_reduced[best_mean] = [best_amplitude, best_pvalue]
                best_mean = current_mean
                best_amplitude = value[0]
                best_pvalue = pvalue
        
        return fitting_results_reduced


    def adjusted_r_squared(self, peaks, fitting_results):  
        masses = peaks[:,0]
        y_true = peaks[:,1]
        mean = np.array(list(fitting_results.keys()))
        amplitude = np.array(list(fitting_results.values()))[:,0]
        y_pred = multi_gaussian(masses, amplitude, mean, self.stddev)
        r_square_value = r2_score(y_true, y_pred) 
        number_data_points = len(peaks) 
        number_variables = 1 + len(fitting_results)*2
        adjusted_r_squared = 1.0 - ( (1.0-r_square_value)*(number_data_points-1.0)/(number_data_points-number_variables-1.0) )
        return adjusted_r_squared
    
    
    def determine_unmodified_species_mass(self, fitting_results, species_mass, mass_tolerance):
        detected_mass_shifts = np.array(list(fitting_results.keys()))
        masses_in_interval = detected_mass_shifts[(detected_mass_shifts>=species_mass-mass_tolerance) & (detected_mass_shifts<=species_mass+mass_tolerance)]
        if len(masses_in_interval) > 1:
            closest_to_species_mass_ix = abs(masses_in_interval - species_mass).argmin()
            unmodified_species_mass = masses_in_interval[closest_to_species_mass_ix]
        elif len(masses_in_interval) == 1:
            unmodified_species_mass = masses_in_interval[0]
            
        else:
            unmodified_species_mass = 0
        
        return unmodified_species_mass
    
    
    def refit_amplitudes(self, peaks, fitting_results, sn_threshold):
        fitting_results_refitted = {}
        masses = peaks[:,0]
        intensities = peaks[:,1]
        mean = np.array(list(fitting_results.keys()))
        amplitude = np.array(list(fitting_results.values()))[:,0]
        refitted_amplitudes = optimize.least_squares(self.__error_func, bounds=(0, np.inf),
                                                     x0=amplitude, args=(mean, self.stddev, masses, intensities))
        ix = 0
        for key, values in fitting_results.items():
            
            if (refitted_amplitudes.x[ix] > sn_threshold) and (refitted_amplitudes.x[ix] > amplitude[ix]*0.5):
                pvalue = values[1]
                fitting_results_refitted[key] = [refitted_amplitudes.x[ix], pvalue]
          
            ix += 1
    
        return fitting_results_refitted

    
    def __error_func(self, amplitude, mean, stddev, x, y):
        return multi_gaussian(x, amplitude, mean, stddev) - y   

    


