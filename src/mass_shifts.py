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
import multiprocessing as mp


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
        self.identified_masses_table.to_csv(path_name+'identified_masses_table.csv', sep=',')    


    def determine_ptm_patterns(self, modifications, maximal_mass_error_list):
        self.ptm_patterns_table = self.__create_ptm_pattern_table(modifications)
        lp_model = LinearProgram()
        lp_model.set_ptm_mass_shifts(modifications.modification_masses)
        lp_model.set_upper_bounds(modifications.upper_bounds)
        minimal_mass_shift = min(np.abs(modifications.modification_masses))
        
        pool = mp.Pool(mp.cpu_count())
        for ix in range(len(self.mass_shifts)):
            mass_shift = self.mass_shifts[ix]
            maximal_mass_error = maximal_mass_error_list[ix]

            if mass_shift >= minimal_mass_shift:
                pool.apply(self.run_lp(), args=(mass_shift, lp_model, maximal_mass_error))

            utils.progress(ix+1, len(self.mass_shifts))

        pool.close()


    def run_lp(self, mass_shift, lp_model, maximal_mass_error):
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
                self.ptm_patterns_table.append(row_entry)



    def estimate_maximal_mass_error(self, mass_error):
        mass_mean = self.identified_masses_table['mass mean'].values
        mass_std = self.identified_masses_table['mass std'].values
        maximal_mass_error = [mass_mean[i]*mass_error*1.0e-6 if np.isnan(mass_std[i]) else mass_std[i] for i in range(len(mass_std))]
        return maximal_mass_error
        
    
    def __create_ptm_pattern_table(self, modifications):
        ptm_column_name_separator = ','
        self.ptm_column_name = ptm_column_name_separator.join(modifications.acronyms)
        column_names = ['mass shift', 'mass error (Da)', 'PTM pattern ('+self.ptm_column_name+')', 'amount of PTMs']
        ptm_pattern_table = pd.DataFrame(columns=column_names) 
        return ptm_pattern_table
    
    
    def add_ptm_patterns_to_table(self):
        by_min_amount_ptms = self.ptm_patterns_table.groupby('mass shift', 'PTM pattern ('+self.ptm_column_name+')')['amount of PTMs'].min()
        
        
        
        
        
    """
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
    """
   
"""
    def save_identified_patterns(self, ptm_patterns, mod, output_name, top=3):
        ptm_column_name_separator = ', '
        ptm_column_name = ptm_column_name_separator.join(mod.acronyms)
        column_names = ['mass shift', 'mass error (Da)', 'PTM pattern ('+ptm_column_name+')', 'amount of PTMs']
        output_df = pd.DataFrame(columns=column_names)
        output_top_df = pd.DataFrame(columns=column_names)
        for key, values in ptm_patterns.items():
            intermediate_df = pd.DataFrame.from_dict(values, orient='index', columns=column_names)
            intermediate_df.sort_values(by=['amount of PTMs', 'mass error (Da)'], key=abs, inplace=True)      
            output_df = output_df.append(intermediate_df)
            output_top_df = output_top_df.append(intermediate_df.iloc[:top,:])

        output_df.reset_index(drop=True, inplace=True)
        output_top_df.reset_index(drop=True, inplace=True)
        output_df.to_csv(output_name+'_ptm_patterns.csv', sep=' ')
        output_top_df.to_csv(output_name+'_ptm_patterns_top_'+ str(top) +'.csv', sep=' ')    
        return output_df, output_top_df


    def add_ptm_patterns_to_table(self, ptm_patterns, output_df, mod):
        ptm_column_name_separator = ', '
        ptm_column_name = ptm_column_name_separator.join(mod.acronyms)
        column_name = 'PTM pattern ('+ptm_column_name+')'
        self.identified_masses_table['best '+column_name] = np.nan
        
        for ix, row in output_df.iterrows():
            row_index = self.identified_masses_table[self.identified_masses_table['mass shift']==row['mass shift']].index.values[0] 
            self.identified_masses_table.loc[row_index,'best '+column_name] = str(row[column_name])

"""