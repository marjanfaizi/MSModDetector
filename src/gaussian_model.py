#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 2 2021

@author: Marjan Faizi
"""

import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import chisquare
import utils
import config

class GaussianModel(object): 
    """
    This class fits a gaussian distribution to a protein"s isotope distribution observed in the mass spectra.
    The standard deviation is given by a function that maps protein mass to the standard deviation of the respective isotope cluster.
    This mapping function is determined by calculating the theoretical isotope distribution of every human proteome.
    The goodness of fit is measured with the chi-squared test: a high p-value indicates that both distributions are similar.
    The dataframe fitting_results stores for every mean:
            - the fitted amplitude
            - the standard deviation (stddev)
            - the corresponding pvalue
            - and the window size for which the best fit was achieved 
            - and the corresponding relative abundance of the peak cluster
    """

    def __init__(self, sample_name, stddev_isotope_distribution):
        self.fitting_results = pd.DataFrame(columns=["sample_name", "mean", "amplitude", "chi_score", "pvalue", "window_size", "relative_abundance"]) 
        self.sample_name = sample_name
        self.stddev = stddev_isotope_distribution
        self.maxfev = 100000
        self.degree_of_freedom = 3
        self.sample_size_threshold = 5
        self.step_size = 1


    def fit_gaussian_within_window(self, peaks, pvalue_threshold):
        masses = peaks[:, 0]
        best_fit = {"fitted_mean": masses[0], "fitted_amplitude": 0, "pvalue": 0, "chi_score": 0, 
                    "window_size":  self.variable_window_sizes[0]}
        window_range_start = masses[0]; window_range_end = masses[0]
        
        while window_range_end <= masses[-1]:
                            
            best_window_size_fit = {"fitted_mean": 0, "fitted_amplitude": 0, "pvalue": 0, "chi_score": 0, 
                                    "window_size": 0}
            for window_size in self.variable_window_sizes:
                peaks_in_window = peaks[(masses >= window_range_start) & (masses <= window_range_start+window_size)]
                if peaks_in_window.size:
                    sample_size = peaks_in_window.shape[0]    
                    if sample_size >= self.sample_size_threshold:
                        optimized_param = self.fit_gaussian(peaks_in_window)
                        fitted_amplitude = optimized_param[0]
                        fitted_mean = optimized_param[1]
                        chi_square_result = self.chi_square_test(peaks_in_window, fitted_amplitude, fitted_mean)
                        pvalue = chi_square_result.pvalue
                        chi_score = chi_square_result.statistic
                        
                        if pvalue > best_window_size_fit["pvalue"]:
                            best_window_size_fit = {"fitted_mean": fitted_mean, "fitted_amplitude": fitted_amplitude, 
                                                    "pvalue": pvalue, "chi_score": chi_score, "window_size": window_size}

            min_distance = (0.5*best_fit["window_size"] + 0.5*best_window_size_fit["window_size"])*config.allowed_overlap_fitting_window
            if np.abs(best_window_size_fit["fitted_mean"]-best_fit["fitted_mean"]) <= min_distance and best_window_size_fit["pvalue"] > best_fit["pvalue"]:
                best_fit = best_window_size_fit.copy()
    
            elif np.abs(best_window_size_fit["fitted_mean"]-best_fit["fitted_mean"]) > min_distance or window_range_start + best_fit["window_size"] >= masses[-1]:
                self.fitting_results = self.fitting_results.append({"sample_name": self.sample_name, 
                                                                    "mean": best_fit["fitted_mean"], 
                                                                    "amplitude": best_fit["fitted_amplitude"], 
                                                                    "pvalue": best_fit["pvalue"],
                                                                    "chi_score": best_fit["chi_score"], 
                                                                    "window_size": best_fit["window_size"]}, ignore_index=True)
                best_fit = best_window_size_fit.copy()


            window_range_start += self.step_size
            window_range_end = window_range_start + best_fit["window_size"]

        self.fitting_results = self.fitting_results[self.fitting_results["pvalue"] >= pvalue_threshold]
        self.fitting_results.reset_index(drop=True, inplace=True)


    def fit_gaussian(self, peaks, mean=None, amplitude=None):
        masses = peaks[:, 0]; intensities = peaks[:, 1]           
        index_most_abundant_peak = intensities.argmax()
        initial_amplitude = intensities[index_most_abundant_peak]
            
        if amplitude==None and mean==None:
            initial_mean = masses[index_most_abundant_peak]
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude, mean: utils.gaussian(x, amplitude, mean, self.stddev), 
                                                    masses, intensities, maxfev=self.maxfev, 
                                                    bounds=([0, masses[0]], [1, masses[-1]]),
                                                    p0=[initial_amplitude, initial_mean])
        elif amplitude==None and mean!=None:
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude: utils.gaussian(x, amplitude, mean, self.stddev), 
                                                    masses, intensities, maxfev=self.maxfev,  bounds=(0, 1),
                                                    p0=[initial_amplitude])
        
        return optimized_param


    def determine_variable_window_sizes(self, mean):
        amplitude = 1
        masses = np.arange(mean-100, mean+100)
        intensities = utils.gaussian(masses, amplitude, mean, self.stddev) 
        most_abundant_masses_lb = masses[intensities > intensities.max()*1.0-config.window_size_lb]
        most_abundant_masses_ub = masses[intensities > intensities.max()*1.0-config.window_size_ub]
        lower_window_size = round((most_abundant_masses_lb[-1]-most_abundant_masses_lb[0]))
        upper_window_size = np.ceil((most_abundant_masses_ub[-1]-most_abundant_masses_ub[0]))
        variable_window_sizes = np.arange(lower_window_size, upper_window_size, 1)
        self.variable_window_sizes = variable_window_sizes


    def chi_square_test(self, selected_region, amplitude, mean):
        observed_masses = selected_region[:,0]; observed_intensities = selected_region[:,1]
        predicted_intensities = utils.gaussian(observed_masses, amplitude, mean, self.stddev)
        chi_square_score = chisquare(observed_intensities, f_exp=predicted_intensities, ddof=self.degree_of_freedom)
        return chi_square_score


    def refit_results(self, peaks, noise_level, refit_mean=False):
        repeat_fitting = 2
        while repeat_fitting > 0:
            if not self.fitting_results.empty:
                masses = peaks[:,0]; intensities = peaks[:,1]
                error_func_amp = lambda amplitude, x, y: (utils.multi_gaussian(x, amplitude, self.fitting_results["mean"].values, self.stddev) - y)**2         
                refitted_amplitudes = optimize.least_squares(error_func_amp, bounds=(0, 1),
                                                             x0=self.fitting_results["amplitude"].values, 
                                                             args=(masses, intensities))
                
                self.fitting_results["amplitude"] = refitted_amplitudes.x
                self.fitting_results = self.fitting_results[self.fitting_results["amplitude"] > noise_level]
                self.fitting_results.reset_index(drop=True, inplace=True)

                if refit_mean:
                    error_func_mean = lambda mean, x, y: (utils.multi_gaussian(x,  self.fitting_results["amplitude"].values, mean, self.stddev) - y)**2
                    delta = 3
                    lb = self.fitting_results["mean"].values-delta
                    ub = self.fitting_results["mean"].values+delta
                    refitted_means = optimize.least_squares(error_func_mean, bounds=(lb, ub),
                                                             x0=self.fitting_results["mean"].values, 
                                                             args=(masses, intensities))

                    self.fitting_results["mean"] = refitted_means.x
        
            repeat_fitting -= 1


    def calculate_relative_abundaces(self, start_mass, end_mass):
        abundances = utils.integral_of_gaussian(self.fitting_results["amplitude"].values, self.stddev)
        total_protein_abundance = sum(abundances)
        self.fitting_results["relative_abundance"] = abundances/total_protein_abundance
        
                

