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
from sklearn.metrics import r2_score
import utils
import myconfig as config

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

    def __init__(self, sample_name):
        self.fitting_results = pd.DataFrame(columns=['sample_name', 'means', 'amplitudes', 'stddevs', 'p-values', 'window_sizes', 'relative_abundances']) 
        self.sample_name = sample_name
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
                          
            stddev = utils.mapping_mass_to_stddev(mass)    
            self.fitting_results = self.fitting_results.append({'sample_name': self.sample_name, 'means': mass, 'amplitudes': best_fitted_amplitude, 
                                                                'stddevs': stddev, 'p-values': best_pvalue, 'window_sizes': best_window_size}, ignore_index=True)


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
        most_abundant_masses_lb = masses[intensities > intensities.max()*1.0-config.window_size_lb]
        most_abundant_masses_ub = masses[intensities > intensities.max()*1.0-config.window_size_ub]
        lower_window_size = round((most_abundant_masses_lb[-1]-most_abundant_masses_lb[0])/2)
        upper_window_size = np.ceil((most_abundant_masses_ub[-1]-most_abundant_masses_ub[0])/2)
        window_sizes = np.arange(lower_window_size, upper_window_size, 1)
        self.fitting_window_sizes = window_sizes


    def chi_square_test(self, selected_region, amplitude, mean):
        observed_masses = selected_region[:,0]; observed_intensities = selected_region[:,1]
        stddev = utils.mapping_mass_to_stddev(mean)
        predicted_intensities = utils.gaussian(observed_masses, amplitude, mean, stddev)
        chi_square_score = chisquare(observed_intensities, f_exp=predicted_intensities, ddof=self.chi_square_dof)
        return chi_square_score.pvalue
    
     
    def filter_fitting_results(self, pvalue_threshold):
        if not self.fitting_results.empty: 
            self.__remove_overlapping_fitting_results()
            self.fitting_results = self.fitting_results[self.fitting_results['p-values'] >= pvalue_threshold]
            self.fitting_results.reset_index(drop=True, inplace=True)


    def __remove_overlapping_fitting_results(self):
        ix_reduced_fitting_results = []
        best_pvalue = 0.0 
        best_index = 0
        last_mean = self.fitting_results.tail(1)['means'].values[0]
        all_window_sizes = np.append(0, self.fitting_results['window_sizes'].values)
        for index, row in self.fitting_results.iterrows():
            window_size = max(all_window_sizes[index], all_window_sizes[index+1])
            if np.abs(self.fitting_results.loc[best_index, 'means']-row['means']) <= window_size and row['p-values'] > best_pvalue:
                best_index = index
                best_pvalue = row['p-values']
            
            elif np.abs(self.fitting_results.loc[best_index, 'means']-row['means']) > window_size:
                ix_reduced_fitting_results.append(best_index)
                best_index = index
                best_pvalue = row['p-values']    
                
                if self.fitting_results.loc[best_index, 'means'] == last_mean:
                    ix_reduced_fitting_results.append(best_index)
                
        self.fitting_results = self.fitting_results.filter(items = ix_reduced_fitting_results, axis=0)
        self.fitting_results.reset_index(drop=True, inplace=True)


    
    def refit_amplitudes(self, peaks, sn_threshold):
        masses = peaks[:,0]; intensities = peaks[:,1]
        if not self.fitting_results.empty: 
            refitted_amplitudes = optimize.least_squares(self.__error_func, bounds=(0, np.inf),
                                                         x0=self.fitting_results.amplitudes.values, 
                                                         args=(self.fitting_results.means.values, self.fitting_results.stddevs.values, masses, intensities))
            ix_reduced_fitting_results = []
            for index, row in self.fitting_results.iterrows():
                if (refitted_amplitudes.x[index] > sn_threshold):
                    ix_reduced_fitting_results.append(index)
        
            self.fitting_results = self.fitting_results.filter(items = ix_reduced_fitting_results, axis=0)
            self.fitting_results.reset_index(drop=True, inplace=True)
            self.fitting_results['amplitudes'] = refitted_amplitudes.x[ix_reduced_fitting_results]

    
    def __error_func(self, amplitude, mean, stddev, x, y):
        return utils.multi_gaussian(x, amplitude, mean, stddev) - y   

    
    def calculate_relative_abundaces(self, start_mass, end_mass):
        x_values = np.arange(start_mass, end_mass)
        total_protein_abundance = np.trapz(utils.multi_gaussian(x_values, self.fitting_results.amplitudes, self.fitting_results.means, 
                                                                self.fitting_results.stddevs), x=x_values)
        relative_abundances = []
        for index, row in self.fitting_results.iterrows():
            species_abundance = np.trapz(utils.gaussian(x_values, row['amplitudes'], row['means'], row['stddevs']), x=x_values)
            relative_abundances.append(species_abundance/total_protein_abundance)

        self.fitting_results['relative_abundances'] = relative_abundances
    
    
    
    