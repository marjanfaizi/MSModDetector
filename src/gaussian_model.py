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
from scipy.stats import wasserstein_distance

class GaussianModel(object): 
    """
    This class fits a gaussian distribution to a protein's isotope distribution observed in the mass spectra.
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
        self.fitting_results = pd.DataFrame(columns=["sample_name", "mean", "amplitude", "chi_score", 
                                                     "pvalue", "window_size", "relative_abundance"]) 
        self.sample_name = sample_name
        self.stddev = stddev_isotope_distribution
        self.maxfev = 100000
        self.degree_of_freedom = 3
        self.sample_size_threshold = 7
        self.intensity_threshold = 1e-10
        self.step_size = 1


    def fit_two_gaussian_within_window(self, peaks, pvalue_threshold, noise_level):
        masses = peaks[:, 0]; intensities = peaks[:, 1]
        best_fit = {"fitted_mean": np.array([masses[0], masses[0]]), "fitted_amplitude": [0,0], "pvalue": 0, "chi_score": 0, 
                    "window_size":  self.variable_window_sizes[0]}
        window_range_start = masses[0]; window_range_end = masses[0]
        
        while window_range_end <= masses[-1]:
            best_window_size_fit = {"fitted_mean": np.array([0,0]), "fitted_amplitude": [0,0], "pvalue": 0, "chi_score": 0, 
                                    "window_size": 0}
            
            for window_size in self.variable_window_sizes:
                peaks_in_window = peaks[(masses >= window_range_start) & (masses <= window_range_start+window_size) &
                                        (intensities > self.intensity_threshold)]
                
                if len(peaks_in_window) >= self.sample_size_threshold:
                    optimized_param = self.fit_gaussian(peaks_in_window, two_gaussians=True)
                    fitted_amp1, fitted_amp2, fitted_mean1, fitted_mean2 = optimized_param
                    centroid = np.mean([fitted_mean1, fitted_mean2])
                    peaks_around_fitted_means = peaks[(masses >= centroid-window_size/2) & (masses <= centroid+window_size/2)]

                    if len(peaks_around_fitted_means[peaks_around_fitted_means[:,1]>noise_level]) >= self.sample_size_threshold:
                        chi_square_result = self.chi_square_test(peaks_around_fitted_means, fitted_amp1, fitted_mean1, fitted_amp2, fitted_mean2)
                        pvalue = chi_square_result.pvalue
                        chi_score = chi_square_result.statistic       
                        
                        if pvalue > best_window_size_fit["pvalue"]:
                            best_window_size_fit = {"fitted_mean": np.array([fitted_mean1, fitted_mean2]), 
                                                    "fitted_amplitude": [fitted_amp1, fitted_amp2], 
                                                    "pvalue": pvalue, "chi_score": chi_score, 
                                                    "window_size": window_size}

            min_distance = (0.5*best_fit["window_size"] + 0.5*best_window_size_fit["window_size"])
            if np.abs(best_window_size_fit["fitted_mean"].mean()-best_fit["fitted_mean"].mean()) <= min_distance and best_window_size_fit["pvalue"] > best_fit["pvalue"]:
                best_fit = best_window_size_fit.copy()

            elif np.abs(best_window_size_fit["fitted_mean"].mean()-best_fit["fitted_mean"].mean()) > min_distance or window_range_start + best_fit["window_size"] >= masses[-1]:
                if best_fit["fitted_mean"].min() == best_fit["fitted_mean"].max():
                    self.__save_fitting_results(best_fit, list_entry=0)               
                else:
                    self.__save_fitting_results(best_fit, list_entry=0)
                    self.__save_fitting_results(best_fit, list_entry=1)
                best_fit = best_window_size_fit.copy()


            window_range_start += self.step_size
            window_range_end = window_range_start + best_fit["window_size"]

        if not best_fit["fitted_mean"][0] in self.fitting_results["mean"] and not best_fit["fitted_mean"][1] in self.fitting_results["mean"]:
            if best_fit["fitted_mean"].min() == best_fit["fitted_mean"].max():
                self.__save_fitting_results(best_fit, list_entry=0)               
            else:
                self.__save_fitting_results(best_fit, list_entry=0)
                self.__save_fitting_results(best_fit, list_entry=1)


        self.fitting_results = self.fitting_results[self.fitting_results["pvalue"] >= pvalue_threshold]
        self.fitting_results.reset_index(drop=True, inplace=True)



    def fit_gaussian_within_window(self, peaks, allowed_overlap_fitting_window, pvalue_threshold, noise_level):
        masses = peaks[:, 0]
        best_fit = {"fitted_mean": masses[0], "fitted_amplitude": 0, "pvalue": 0, "chi_score": 0, 
                    "window_size":  self.variable_window_sizes[0]}
        window_range_start = masses[0]; window_range_end = masses[0]
        
        while window_range_end <= masses[-1]:
                            
            best_window_size_fit = {"fitted_mean": 0, "fitted_amplitude": 0, "pvalue": 0, "chi_score": 0, 
                                     "window_size": 0}
            for window_size in self.variable_window_sizes:
                peaks_in_window = peaks[(masses >= window_range_start) & (masses <= window_range_start+window_size)]
                non_zero_intensity_ix = np.where(peaks_in_window[:, 1] > self.intensity_threshold)[0]
                peaks_in_window = peaks_in_window[non_zero_intensity_ix]
                if len(peaks_in_window) >= self.sample_size_threshold:
                    optimized_param = self.fit_gaussian(peaks_in_window)
                    fitted_amplitude, fitted_mean = optimized_param
                    peaks_around_fitted_mean = peaks[(masses >= fitted_mean-window_size/2) & (masses <= fitted_mean+window_size/2)]

                    if len(peaks_around_fitted_mean[peaks_around_fitted_mean[:,1]>noise_level]) >= self.sample_size_threshold:
                        chi_square_result = self.chi_square_test(peaks_around_fitted_mean, fitted_amplitude, fitted_mean)
                        pvalue = chi_square_result.pvalue; chi_score = chi_square_result.statistic       
                        
                        if pvalue > best_window_size_fit["pvalue"]:
                            best_window_size_fit = {"fitted_mean": fitted_mean, "fitted_amplitude": fitted_amplitude, 
                                                    "pvalue": pvalue, "chi_score": chi_score,
                                                    "window_size": window_size}

            min_distance = (0.5*best_fit["window_size"] + 0.5*best_window_size_fit["window_size"])*allowed_overlap_fitting_window
            if np.abs(best_window_size_fit["fitted_mean"]-best_fit["fitted_mean"]) <= min_distance and best_window_size_fit["pvalue"] > best_fit["pvalue"]:
                best_fit = best_window_size_fit.copy()
    
            elif np.abs(best_window_size_fit["fitted_mean"]-best_fit["fitted_mean"]) > min_distance or window_range_start + best_fit["window_size"] >= masses[-1]:
                self.__save_fitting_results(best_fit)
                best_fit = best_window_size_fit.copy()


            window_range_start += self.step_size
            window_range_end = window_range_start + best_fit["window_size"]

        if not best_fit["fitted_mean"] in self.fitting_results["mean"]:
            self.__save_fitting_results(best_fit)

        self.fitting_results = self.fitting_results[self.fitting_results["pvalue"] >= pvalue_threshold]
        self.fitting_results.reset_index(drop=True, inplace=True)


    def __save_fitting_results(self, fit_dict, list_entry=None):
        if list_entry==None: 
            self.fitting_results = self.fitting_results.append({"sample_name": self.sample_name, 
                                                                "mean": fit_dict["fitted_mean"], 
                                                                "amplitude": fit_dict["fitted_amplitude"], 
                                                                "pvalue": fit_dict["pvalue"],
                                                                "chi_score": fit_dict["chi_score"], 
                                                                "window_size": fit_dict["window_size"]}, ignore_index=True)
        else:
            self.fitting_results = self.fitting_results.append({"sample_name": self.sample_name, 
                                                                "mean": fit_dict["fitted_mean"][list_entry], 
                                                                "amplitude": fit_dict["fitted_amplitude"][list_entry], 
                                                                "pvalue": fit_dict["pvalue"],
                                                                "chi_score": fit_dict["chi_score"], 
                                                                "window_size": fit_dict["window_size"]}, ignore_index=True)



    def fit_gaussian(self, peaks, mean=None, amplitude=None, two_gaussians=False):
        masses = peaks[:, 0]; intensities = peaks[:, 1]           
        index_most_abundant_peak = intensities.argmax()
        initial_amplitude = intensities[index_most_abundant_peak]
        initial_mean = masses[index_most_abundant_peak]

        if amplitude==None and mean==None:
            if not two_gaussians:
                optimized_param, _ = optimize.curve_fit(lambda x, amplitude, mean: utils.gaussian(x, amplitude, mean, self.stddev), 
                                                        masses, intensities, maxfev=self.maxfev, 
                                                        bounds=([0, masses[0]], [1, masses[-1]]),
                                                        p0=[initial_amplitude, initial_mean])        
            else:
                optimized_param, _ = optimize.curve_fit(lambda x, amp1, amp2, mean1, mean2: utils.two_gaussian(x, amp1, amp2, mean1, mean2, self.stddev), 
                                                        masses, intensities, maxfev=self.maxfev, 
                                                        bounds=([0, 0, masses[0], masses[0]], [1, 1, masses[-1], masses[-1]]),
                                                        p0=[initial_amplitude, initial_amplitude, initial_mean, initial_mean])
        elif amplitude==None and mean!=None:
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude: utils.gaussian(x, amplitude, mean, self.stddev), 
                                                    masses, intensities, maxfev=self.maxfev,  bounds=(0, 1),
                                                    p0=[initial_amplitude])
        
        return optimized_param


    def determine_variable_window_sizes(self, mean, window_size_lb, window_size_ub):
        amplitude = 1
        masses = np.arange(mean-100, mean+100)
        intensities = utils.gaussian(masses, amplitude, mean, self.stddev) 
        most_abundant_masses_lb = masses[intensities > intensities.max()*1.0-window_size_lb]
        most_abundant_masses_ub = masses[intensities > intensities.max()*1.0-window_size_ub]
        lower_window_size = round((most_abundant_masses_lb[-1]-most_abundant_masses_lb[0]))
        upper_window_size = np.ceil((most_abundant_masses_ub[-1]-most_abundant_masses_ub[0]))
        variable_window_sizes = np.arange(lower_window_size, upper_window_size, 1)
        self.variable_window_sizes = variable_window_sizes


    def chi_square_test(self, selected_region, amplitude, mean, amplitude2=None, mean2=None):
        observed_masses = selected_region[:,0]; observed_intensities = selected_region[:,1]
        if amplitude2 == None:
            predicted_intensities = utils.gaussian(observed_masses, amplitude, mean, self.stddev)
        else:
            predicted_intensities = utils.two_gaussian(observed_masses, amplitude, amplitude2, mean, mean2, self.stddev)
        chi_square_score = chisquare(observed_intensities, f_exp=predicted_intensities, ddof=self.degree_of_freedom)
        return chi_square_score


    def refit_results(self, peaks, noise_level, refit_mean=False):
        repeat_fitting = 2
        while repeat_fitting > 0:
            if not self.fitting_results.empty:
                masses = peaks[:,0]; intensities = peaks[:,1]
                error_func_amp = lambda amplitude, x, y: utils.multi_gaussian(x, amplitude, self.fitting_results["mean"].values, self.stddev) - y    
                refitted_amplitudes = optimize.least_squares(error_func_amp, bounds=(0, 1), loss="soft_l1", 
                                                             x0=self.fitting_results["amplitude"].values, 
                                                             args=(masses, intensities))
                
                self.fitting_results["amplitude"] = refitted_amplitudes.x
                self.fitting_results = self.fitting_results[self.fitting_results["amplitude"] > noise_level]
                self.fitting_results.reset_index(drop=True, inplace=True)

                if refit_mean:
                    error_func_mean = lambda mean, x, y: utils.multi_gaussian(x, self.fitting_results["amplitude"].values, mean, self.stddev) - y
                    delta = 3
                    lb = self.fitting_results["mean"].values-delta
                    ub = self.fitting_results["mean"].values+delta
                    refitted_means = optimize.least_squares(error_func_mean, bounds=(lb, ub), loss="soft_l1",
                                                             x0=self.fitting_results["mean"].values, 
                                                             args=(masses, intensities))

                    self.fitting_results["mean"] = refitted_means.x
        
            repeat_fitting -= 1



    def calculate_relative_abundaces(self, start_mass, end_mass):
        abundances = utils.integral_of_gaussian(self.fitting_results["amplitude"].values, self.stddev)
        total_protein_abundance = sum(abundances)
        self.total_protein_abundance = total_protein_abundance
        self.fitting_results["relative_abundance"] = abundances/total_protein_abundance
        
                

