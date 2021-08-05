#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 6 2021

@author: Marjan Faizi
"""

import numpy as np
import sys
#from scipy.optimize import least_squares
#from fit_gaussian_model import multi_gaussian
from linear_program import LinearProgram



class MassShifts(object): 
    
    def __init__(self, mass_shifts):
        self.mean = np.array(list(mass_shifts.keys()))
        self.amplitude = np.array(list(mass_shifts.values()))[:,0]
        self.pvalue = np.array(list(mass_shifts.values()))[:,1]
        self.mass_shifts = np.zeros(len(mass_shifts))
        self.unmodified_species_mass = 0.0
     
        
    def set_unmodified_species_mass(self, unmodified_species_mass):
        self.unmodified_species_mass = unmodified_species_mass
     
    
    def calculate_mass_shifts(self):
        mass_shift_threshold = -0.1
        all_mass_shifts = self.mean - self.unmodified_species_mass     
        mask = all_mass_shifts > mass_shift_threshold
        self.mass_shifts = all_mass_shifts[mask]
        self.mean = self.mean[mask]
        self.amplitude = self.amplitude[mask]
        self.pvalue = self.pvalue[mask]

    """
    def refit_amplitudes(self, peaks, stddev, sn_threshold):
        masses = peaks[:,0]
        intensities = peaks[:,1]
        refitted_amplitudes = least_squares(self.__error_func, bounds=(0, np.inf),
                                            x0=self.amplitude, args=(self.mean, stddev, masses, intensities))
        self.__remove_mass_shifts(sn_threshold, refitted_amplitudes.x)
        refitted_amplitudes2 = least_squares(self.__error_func, bounds=(0, np.inf),
                                            x0=self.amplitude, args=(self.mean, stddev, masses, intensities))
        self.amplitude = refitted_amplitudes2.x

    
    def __error_func(self, amplitude, mean, stddev, x, y):
        return multi_gaussian(x, amplitude, mean, stddev) - y

    
    def __remove_mass_shifts(self, sn_threshold, refitted_amplitudes):
        mass_shifts_below_sn_ix = np.where((refitted_amplitudes <= sn_threshold) | (refitted_amplitudes < self.amplitude*0.5))[0]
        self.mean = np.delete(self.mean, mass_shifts_below_sn_ix)
        self.amplitude = np.delete(self.amplitude, mass_shifts_below_sn_ix)
        self.pvalue = np.delete(self.pvalue, mass_shifts_below_sn_ix)
        self.mass_shifts = np.delete(self.mass_shifts, mass_shifts_below_sn_ix)
   """
  

    def determine_all_ptm_patterns(self, mass_tolerance, modifications):
        lp_model = LinearProgram()
        lp_model.set_ptm_mass_shifts(modifications.modification_masses)
        lp_model.set_upper_bounds(modifications.upper_bounds)
        minimal_mass_shift = min(np.abs(modifications.modification_masses))
        all_ptm_patterns = {}
        for ix in range(len(self.mass_shifts)):
            maximal_mass_error = self.mean[ix]*mass_tolerance*1.0e-6
            if (self.mass_shifts[ix] >= minimal_mass_shift):
                ptm_patterns = self.estimate_ptm_patterns(self.mass_shifts[ix], lp_model, maximal_mass_error)
                if len(ptm_patterns) > 0:
                    all_ptm_patterns[ix] = ptm_patterns

            self.__progress(ix+1, len(self.mass_shifts))
        return all_ptm_patterns


    def determine_all_ptm_patterns2(self, maximal_mass_error, modifications, mass_shifts):
        lp_model = LinearProgram()
        lp_model.set_ptm_mass_shifts(modifications.modification_masses)
        lp_model.set_upper_bounds(modifications.upper_bounds)
        minimal_mass_shift = min(np.abs(modifications.modification_masses))
        all_ptm_patterns = {}
        for ix in range(len(mass_shifts)):
            if (mass_shifts[ix] >= minimal_mass_shift):
                ptm_patterns = self.estimate_ptm_patterns(mass_shifts[ix], lp_model, maximal_mass_error[ix])
                if len(ptm_patterns) > 0:
                    all_ptm_patterns[mass_shifts[ix]] = ptm_patterns

            self.__progress(ix+1, len(mass_shifts))
        return all_ptm_patterns


    def estimate_ptm_patterns(self, mass_shift, lp_model, maximal_mass_error):
        ptm_patterns = {}
        ix = 0
        mass_error = 0.0     
        lp_model.reset_mass_error()
        lp_model.set_observed_mass_shift(mass_shift) 
        while abs(mass_error) < maximal_mass_error:            
            lp_model.solve_linear_program_for_objective_with_absolute_value()
            mass_error = lp_model.get_objective_value()
            lp_model.set_minimal_mass_error(abs(mass_error))
            pattern = lp_model.get_x_values()
            number_ptm_types = np.array(pattern).sum()
            if abs(mass_error) < maximal_mass_error:
                ptm_patterns[ix] = [mass_shift, mass_error, pattern, number_ptm_types]
                ix += 1

        return ptm_patterns

   
    def __progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush() 
