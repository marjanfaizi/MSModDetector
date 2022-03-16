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
import sys

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
        self.fitting_results = pd.DataFrame(columns=["sample_name", "mean", "amplitude", "chi_score", "p_value", "window_size", "relative_abundance"]) 
        self.sample_name = sample_name
        self.stddev = stddev_isotope_distribution
        self.maxfev = 100000
        self.degree_of_freedom = 3
        self.sample_size_threshold = 5


    def fit_gaussian_to_single_peaks(self, peaks, noise_level, pvalue_threshold):
        for peak in peaks:   
            mass = peak[0]
            intensity = peak[1]
            if intensity > noise_level:  

                for window_size in self.fitting_window_sizes:                    
                    selected_region_ix = np.argwhere((peaks[:,0]<=mass+window_size) & (peaks[:,0]>=mass-window_size))[:,0]
                    selected_region = peaks[selected_region_ix]
                    sample_size = selected_region.shape[0]          
                
                    if sample_size >= self.sample_size_threshold:
                        fitted_amplitude = self.fit_gaussian(selected_region, mean=mass)[0]
                        chi_square_result = self.chi_square_test(selected_region, fitted_amplitude, mass)
                        pvalue = chi_square_result.pvalue
                        chi_score = chi_square_result.statistic

                        self.fitting_results = self.fitting_results.append({"sample_name": self.sample_name, "mean": mass, "amplitude": fitted_amplitude, 
                                                                            "chi_score": chi_score, "p_value": pvalue, "window_size": window_size}, ignore_index=True)

        idxmax_pvalue = self.fitting_results.groupby("mean")["p_value"].idxmax().tolist()
        
        if idxmax_pvalue: 
            self.fitting_results = self.fitting_results.iloc[idxmax_pvalue]
            self.fitting_results = self.fitting_results[self.fitting_results["p_value"] >= pvalue_threshold]
            self.fitting_results.reset_index(drop=True, inplace=True)
         
        else:
            print("\nNo peaks found above the pvalue threshold.\n")
            sys.exit()


    def fit_gaussian(self, peaks, mean=None, amplitude=None):
        masses = peaks[:, 0]; intensities = peaks[:, 1]           
        index_most_abundant_peak = intensities.argmax()
        initial_amplitude = intensities[index_most_abundant_peak]
            
        if amplitude==None and mean==None:
            initial_mean = masses[index_most_abundant_peak]
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude, mean: utils.gaussian(x, amplitude, mean, self.stddev), 
                                                    masses, intensities, maxfev=self.maxfev,
                                                    p0=[initial_amplitude, initial_mean])
        elif amplitude==None and mean!=None:
            optimized_param, _ = optimize.curve_fit(lambda x, amplitude: utils.gaussian(x, amplitude, mean, self.stddev), 
                                                    masses, intensities, maxfev=self.maxfev,
                                                    p0=[initial_amplitude])
        
        return optimized_param


    def determine_adaptive_window_sizes(self, mean):
        amplitude = 1
        masses = np.arange(mean-100, mean+100)
        intensities = utils.gaussian(masses, amplitude, mean, self.stddev) 
        most_abundant_masses_lb = masses[intensities > intensities.max()*1.0-config.window_size_lb]
        most_abundant_masses_ub = masses[intensities > intensities.max()*1.0-config.window_size_ub]
        lower_window_size = round((most_abundant_masses_lb[-1]-most_abundant_masses_lb[0])/2)
        upper_window_size = np.ceil((most_abundant_masses_ub[-1]-most_abundant_masses_ub[0])/2)
        window_sizes = np.arange(lower_window_size, upper_window_size, 1)
        self.fitting_window_sizes = window_sizes


    def chi_square_test(self, selected_region, amplitude, mean):
        observed_masses = selected_region[:,0]; observed_intensities = selected_region[:,1]
        predicted_intensities = utils.gaussian(observed_masses, amplitude, mean, self.stddev)
        chi_square_score = chisquare(observed_intensities, f_exp=predicted_intensities, ddof=self.degree_of_freedom)
        return chi_square_score


    def remove_overlapping_fitting_results(self):
        ix_keep_fitting_results = []        
        best_result_ix = 0
        best_pvalue = 0.0 
        last_mean = self.fitting_results.tail(1)["mean"].values[0]
        all_window_sizes = np.append(0, self.fitting_results["window_size"].values)

        for index, row in self.fitting_results.iterrows():
            
            #window_size = max(all_window_sizes[index], all_window_sizes[index+1])
            window_size_sum = all_window_sizes[index] + all_window_sizes[index+1]
            
            if np.abs(self.fitting_results.loc[best_result_ix, "mean"]-row["mean"]) <= (1.0-config.allowed_overlap)*window_size_sum and row["p_value"] > best_pvalue:
                best_result_ix = index
                best_pvalue = row["p_value"]
            
            elif np.abs(self.fitting_results.loc[best_result_ix, "mean"]-row["mean"]) > (1.0-config.allowed_overlap)*window_size_sum:
                ix_keep_fitting_results.append(best_result_ix)
                best_result_ix = index
                best_pvalue = row["p_value"]    
                
                if self.fitting_results.loc[best_result_ix, "mean"] == last_mean:
                    ix_keep_fitting_results.append(best_result_ix)

        self.fitting_results = self.fitting_results.filter(items = ix_keep_fitting_results, axis=0)
        self.fitting_results.reset_index(drop=True, inplace=True)


    
    def refit_amplitudes(self, peaks, noise_level):
        masses = peaks[:,0]; intensities = peaks[:,1]
        if not self.fitting_results.empty:
            refitted_amplitudes = optimize.least_squares(self.__error_func, bounds=(0, np.inf),
                                                         x0=self.fitting_results["amplitude"].values, 
                                                         args=(self.fitting_results["mean"].values, self.stddev, masses, intensities))
            ix_reduced_fitting_results = []
            for index, row in self.fitting_results.iterrows():
                if (refitted_amplitudes.x[index] > noise_level):
                    ix_reduced_fitting_results.append(index)
        
            self.fitting_results = self.fitting_results.filter(items = ix_reduced_fitting_results, axis=0)
            self.fitting_results.reset_index(drop=True, inplace=True)
            self.fitting_results["amplitude"] = refitted_amplitudes.x[ix_reduced_fitting_results]

    
    def __error_func(self, amplitude, mean, stddev, x, y):
        return utils.multi_gaussian(x, amplitude, mean, stddev) - y   

    
    def calculate_relative_abundaces(self, start_mass, end_mass):
        abundances = utils.integral_of_gaussian(self.fitting_results["amplitude"].values, self.stddev)
        total_protein_abundance = sum(abundances)
        self.fitting_results["relative_abundance"] = abundances/total_protein_abundance



