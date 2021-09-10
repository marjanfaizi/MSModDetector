#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 6 2021

@author: Marjan Faizi
"""

import pandas as pd
import numpy as np
from linear_program import LinearProgram
import utils


class MassShifts(object): 
    """
    This class takes the results from fitting gaussian distributions to the observed isotope distributuons and calculates the mass shifts.
    For calculating the mass shifts the mass of an unmodified species has to be determined first.
    For a given mass shift the linear program determines global PTM patterns/combinations and reports the combination with the least amount of PTMs.
    """
    
    def __init__(self):
        self.identified_masses_table = pd.DataFrame(columns=['mass mean', 'mass std']) 
        self.mass_shifts = []
        self.ptm_column_name = ''


    def create_table_identified_masses(self, means, column_name, bin_size_identified_masses):
        if self.identified_masses_table.empty:
            self.identified_masses_table[column_name] = means
        else:
            self.identified_masses_table[column_name] = np.nan
            for mass in means:
                row_index = self.identified_masses_table[(self.identified_masses_table['mass mean']<=mass+bin_size_identified_masses) & 
                                                         (self.identified_masses_table['mass mean']>=mass-bin_size_identified_masses)].index.values
                if len(row_index) > 0:
                    self.identified_masses_table.loc[row_index[0]][column_name] = mass
                else:
                    self.identified_masses_table = self.identified_masses_table.append({column_name: mass}, ignore_index=True)
        
        self.identified_masses_table['mass mean'] = self.identified_masses_table.iloc[:,2:].mean(axis=1).values
        self.identified_masses_table['mass std'] = self.identified_masses_table.iloc[:,2:].std(axis=1).values
        self.identified_masses_table.sort_values(by='mass mean', inplace=True)


    def add_mass_shifts(self, unmodified_species_mass):
        if unmodified_species_mass == 0:
            self.identified_masses_table['mass shift'] = self.identified_masses_table['mass mean'] - self.identified_masses_table['mass mean'].iloc[0] 
        else:
            self.identified_masses_table['mass shift'] = self.identified_masses_table['mass mean'] - unmodified_species_mass  
        
        self.identified_masses_table = self.identified_masses_table[self.identified_masses_table['mass shift'] >= 0]
        self.mass_shifts = self.identified_masses_table['mass shift'].values
        
        
    def save_table_identified_masses(self, path_name):
        self.identified_masses_table = self.identified_masses_table.sort_index(axis=1)
        self.identified_masses_table.to_csv(path_name+'identified_masses_table.csv', sep=',', index=False)    


    def determine_ptm_patterns(self, modifications, maximal_mass_error_array):
        self.ptm_patterns_table = self.__create_ptm_pattern_table(modifications)
        lp_model = LinearProgram()
        lp_model.set_ptm_mass_shifts(modifications.modification_masses)
        lp_model.set_upper_bounds(modifications.upper_bounds)
        minimal_mass_shift = min(np.abs(modifications.modification_masses))

        for ix in range(len(self.mass_shifts)):
            mass_shift = self.mass_shifts[ix]
            maximal_mass_error = maximal_mass_error_array[ix]

            if mass_shift >= minimal_mass_shift:
                mass_error = 0.0     
                lp_model.reset_mass_error()
                lp_model.set_observed_mass_shift(mass_shift) 
                while abs(mass_error) < maximal_mass_error:            
                    lp_model.solve_linear_program_for_objective_with_absolute_value()
                    mass_error = lp_model.get_objective_value()
                    lp_model.set_minimal_mass_error(abs(mass_error))
                    ptm_pattern = lp_model.get_x_values()
                    number_ptm_types = np.array(ptm_pattern).sum()
                    if abs(mass_error) < maximal_mass_error:
                        row_entry = [mass_shift, mass_error, ptm_pattern, number_ptm_types]
                        row_entry_as_series = pd.Series(row_entry, index = self.ptm_patterns_table.columns)
                        self.ptm_patterns_table = self.ptm_patterns_table.append(row_entry_as_series, ignore_index=True)

            utils.progress(ix+1, len(self.mass_shifts))


    def estimate_maximal_mass_error(self, mass_error):
        mass_mean = self.identified_masses_table['mass mean'].values
        mass_std = self.identified_masses_table['mass std'].values
        maximal_mass_error = [mass_mean[i]*mass_error*1.0e-6 if np.isnan(mass_std[i]) else mass_std[i] for i in range(len(mass_std))]
        return np.array(maximal_mass_error)
        
    
    def __create_ptm_pattern_table(self, modifications):
        ptm_column_name_separator = ','
        self.ptm_column_name = ptm_column_name_separator.join(modifications.acronyms)
        column_names = ['mass shift', 'mass error (Da)', 'PTM pattern ('+self.ptm_column_name+')', 'amount of PTMs']
        ptm_pattern_table = pd.DataFrame(columns=column_names) 
        return ptm_pattern_table
    
    
    def add_ptm_patterns_to_table(self):
        self.ptm_patterns_table['amount of PTMs'] = pd.to_numeric(self.ptm_patterns_table['amount of PTMs'])
        best_ptm_patterns = self.ptm_patterns_table.loc[self.ptm_patterns_table.groupby('mass shift')['amount of PTMs'].idxmin()]
        self.identified_masses_table = pd.merge(best_ptm_patterns[['mass shift', 'PTM pattern ('+self.ptm_column_name+')']], 
                                                self.identified_masses_table, how='outer', left_on=['mass shift'], 
                                                right_on=['mass shift'])
        self.identified_masses_table.sort_values(by=['mass shift'], inplace=True)

