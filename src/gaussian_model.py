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
import utils


class GaussianModel(object): 
    """
    This class fits a gaussian distribution to a protein's isotope distribution observed in the mass spectra.
    The standard deviation is given by a function that maps protein mass to the standard deviation of the respective isotope cluster.
    This mapping function is determined by calculating the theoretical isotope distribution of every human proteome.
    The goodness of fit is measured with the chi-squared test: a high p-value indicates that both distributions are similar.
    The dictionary fitting_results stores for every mean:
            - the fitted amplitude
            - the corresponding pvalue
            - and the window size for which the best fit was achieved 
    """

    def __init__(self):
        self.fitting_results = {}
        self.means = []
        self.amplitudes = []
        self.maxfev = 100000
        self.chi_square_dof = 1
        self.sample_size_threshold = 5


    def fit_gaussian_to_single_peaks(self, all_peaks, peaks_above_sn):
        all_masses = all_peaks[:,0]
        masses_above_sn = peaks_above_sn[:,0]

        for mass in masses_above_sn:      
            best_pvalue = -1.0
            
            for window_size in self.fitting_window_sizes:
                selected_region_ix = np.argwhere((all_masses<=mass+window_size) & (all_masses>=mass-window_size))[:,0]
                selected_region = all_peaks[selected_region_ix]
                sample_size = selected_region.shape[0]          
                
                if sample_size >= self.sample_size_threshold:
                    fitted_amplitude = self.fit_gaussian(selected_region, mean=mass)[0]        
                    pvalue = self.chi_square_test(selected_region, fitted_amplitude, mass)
                else:
                    pvalue = 0.0
                    fitted_amplitude = 0.0
    
                if pvalue > best_pvalue:
                    best_pvalue = pvalue
                    best_fitted_amplitude = fitted_amplitude
                    best_window_size = window_size
                          
            self.fitting_results[mass] = [best_fitted_amplitude, best_pvalue, best_window_size]
        self.means = np.array(list(self.fitting_results.keys()))
        self.amplitudes = np.array(list(self.fitting_results.values()))[:,0]


    def fit_gaussian(self, peaks, mean=None, amplitude=None):
        masses = peaks[:, 0]; intensities = peaks[:, 1]           
        index_most_abundant_peak = intensities.argmax()
        initial_amplitude = intensities[index_most_abundant_peak]
            
        if amplitude==None and mean==None:
            initial_mean = masses[index_most_abundant_peak]
            stddev = utils.mapping_mass_to_stddev(initial_mean)
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude, mean: utils.gaussian(x, amplitude, mean, stddev), 
                                                    masses, intensities, maxfev=self.maxfev,
                                                    p0=[initial_amplitude, initial_mean])
        elif amplitude==None and mean!=None:
            stddev = utils.mapping_mass_to_stddev(mean)
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude: utils.gaussian(x, amplitude, mean, stddev), 
                                                    masses, intensities, maxfev=self.maxfev,
                                                    p0=[initial_amplitude])
        
        return optimized_param
     
    
    def determine_adaptive_window_sizes(self, mean):
        amplitude = 1
        masses = np.arange(mean-100, mean+100)
        stddev = utils.mapping_mass_to_stddev(mean)
        intensities = utils.gaussian(masses, amplitude, mean, stddev) 
        most_abundant_50percent_masses = masses[intensities > intensities.max()*0.5]
        most_abundant_90percent_masses = masses[intensities > intensities.max()*0.1]
        lower_window_size = round((most_abundant_50percent_masses[-1]-most_abundant_50percent_masses[0])/2)
        upper_window_size = np.ceil((most_abundant_90percent_masses[-1]-most_abundant_90percent_masses[0])/2)
        window_sizes = np.arange(lower_window_size, upper_window_size, 1)
        self.fitting_window_sizes = window_sizes


    def chi_square_test(self, selected_region, amplitude, mean):
        observed_masses = selected_region[:,0]; observed_intensities = selected_region[:,1]
        stddev = utils.mapping_mass_to_stddev(mean)
        predicted_intensities = utils.gaussian(observed_masses, amplitude, mean, stddev)
        chi_square_score = chisquare(observed_intensities, f_exp=predicted_intensities, ddof=self.chi_square_dof)
        return chi_square_score.pvalue
    
     
    def filter_fitting_results(self, pvalue_threshold):
        self.__select_best_pvalues(pvalue_threshold)
        if len(self.fitting_results) > 1:
            self.__remove_overlapping_fitting_results()


    def __select_best_pvalues(self, pvalue_threshold):
        self.fitting_results = dict(filter(lambda elem: elem[1][1] >= pvalue_threshold, self.fitting_results.items()))
        self.__check_if_dict_empty()


    def __remove_overlapping_fitting_results(self):
        fitting_results_reduced = {}
        best_pvalue = 0.0 
        best_mean = self.means[0]   
        last_mean = self.means[-1]   
        all_window_sizes = np.append(0, np.array(list(self.fitting_results.values()))[:,2])
        ix_window_size = 0

        for current_mean, value in self.fitting_results.items():
            pvalue = value[1]
            window_size = max(all_window_sizes[ix_window_size], all_window_sizes[ix_window_size+1])
            ix_window_size += 1
            
            if np.abs(best_mean-current_mean) <= window_size and pvalue > best_pvalue:
                best_mean = current_mean
                best_amplitude = value[0]
                best_pvalue = pvalue
            
            elif np.abs(best_mean-current_mean) > window_size:
                fitting_results_reduced[best_mean] = [best_amplitude, best_pvalue]
                best_mean = current_mean
                best_amplitude = value[0]
                best_pvalue = pvalue    
                
                if best_mean == last_mean:
                    fitting_results_reduced[best_mean] = [best_amplitude, best_pvalue]

        self.fitting_results = fitting_results_reduced
        self.__check_if_dict_empty()


    def adjusted_r_squared(self, peaks):  
        masses = peaks[:,0]; y_true = peaks[:,1]
        stddev = utils.mapping_mass_to_stddev(self.means)
        y_pred = utils.multi_gaussian(masses, self.amplitudes, self.means, stddev)
        r_square_value = r2_score(y_true, y_pred) 
        number_data_points = len(peaks) 
        number_variables = len(self.fitting_results)*3
        adjusted_r_squared = 1.0 - ( (1.0-r_square_value)*(number_data_points-1.0)/(number_data_points-number_variables-1.0) )
        return adjusted_r_squared
        
    
    def refit_amplitudes(self, peaks, sn_threshold):
        is_empty = self.__check_if_dict_empty()
        if not is_empty:
            fitting_results_refitted = {}
            masses = peaks[:,0]; intensities = peaks[:,1]
            stddev = utils.mapping_mass_to_stddev(self.means)
            refitted_amplitudes = optimize.least_squares(self.__error_func, bounds=(0, np.inf),
                                                         x0=self.amplitudes, args=(self.means, stddev, masses, intensities))
            ix = 0
            for key, values in self.fitting_results.items():
                
                if (refitted_amplitudes.x[ix] > sn_threshold) and (refitted_amplitudes.x[ix] > self.amplitudes[ix]*0.5):
                    pvalue = values[1]
                    fitting_results_refitted[key] = [refitted_amplitudes.x[ix], pvalue]
              
                ix += 1
        
            self.fitting_results = fitting_results_refitted
            self.__check_if_dict_empty()


    def __check_if_dict_empty(self):
        if not self.fitting_results:
            self.means = []
            self.amplitudes = []
            return True
        else:
            self.means = np.array(list(self.fitting_results.keys()))
            self.amplitudes = np.array(list(self.fitting_results.values()))[:,0]
            return False

    
    def __error_func(self, amplitude, mean, stddev, x, y):
        return utils.multi_gaussian(x, amplitude, mean, stddev) - y   

   
