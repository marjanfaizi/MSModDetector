#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 6 2021

@author: Marjan Faizi
"""

import pandas as pd
import numpy as np
#from linear_program import LinearProgram
from linear_program_cvxopt import LinearProgramCVXOPT
import utils
import myconfig as config


class MassShifts(object): 
    """
    This class takes the results from fitting gaussian distributions to the observed isotope distributuons and calculates the mass shifts.
    For calculating the mass shifts the mass of an unmodified species has to be determined first.
    For a given mass shift the linear program determines global PTM patterns/combinations and reports the combination with the least amount of PTMs.
    """
    
    def __init__(self):
        self.identified_masses_table = pd.DataFrame(columns=['mass_index', 'average mass', 'mass std']) 
        self.identified_masses_table['mass_index'] = np.arange(int(config.start_mass_range), int(config.start_mass_range+config.max_mass_shift+20))
        self.mass_shifts = []
        self.laps_run_lp = 5
        self.ptm_column_name = ''


    def create_table_identified_masses(self, means, relative_abundances, column_name):
        self.identified_masses_table['masses '+column_name] = np.nan
        self.identified_masses_table['rel. abundances '+column_name] = np.nan
        for ix in range(len(means)):
            mass = means[ix]             
            abundance = relative_abundances[ix]
            row_index = self.identified_masses_table[self.identified_masses_table['mass_index']==round(mass)]['mass_index'].index.values
            self.identified_masses_table.at[row_index[0], 'masses '+column_name] = mass
            self.identified_masses_table.at[row_index[0], 'rel. abundances '+column_name] = abundance

        self.identified_masses_table['average mass'] = self.identified_masses_table.filter(regex='masses').mean(axis=1).values
        self.identified_masses_table['mass std'] = self.identified_masses_table.filter(regex='masses').std(axis=1).values
        self.identified_masses_table.sort_values(by='average mass', inplace=True)
        

    def set_index(self):
        self.identified_masses_table.set_index('mass_index', inplace=True)  
        
        
# =============================================================================
#     def add_mass_shifts(self, column_names):
#         mass_shift_list = []
#         for name in column_names: 
#             masses = self.identified_masses_table['masses '+name].values
#             unmodified_species_mass = utils.determine_unmodified_species_mass(masses, config.unmodified_species_mass_init, config.unmodified_species_mass_tol) 
#             mass_shifts = masses - unmodified_species_mass
#             mass_shift_list.append(mass_shifts)
#         
#         mean_mass_shifts = np.nanmean(np.array(mass_shift_list), axis=0)
#         self.identified_masses_table['mass shift'] = mean_mass_shifts     
#         self.identified_masses_table = self.identified_masses_table[self.identified_masses_table['mass shift'] >= 0]
#         self.mass_shifts = self.identified_masses_table['mass shift'].values
# =============================================================================

    def add_mass_shifts(self):
        masses = self.identified_masses_table['average mass'].values
        unmodified_species_mass = utils.determine_unmodified_species_mass(masses, config.unmodified_species_mass_init, config.unmodified_species_mass_tol)
        if unmodified_species_mass == 0:
            self.identified_masses_table['mass shift'] = self.identified_masses_table['average mass'] - self.identified_masses_table['average mass'].iloc[0] 
        else:
            self.identified_masses_table['mass shift'] = self.identified_masses_table['average mass'] - unmodified_species_mass  
        
        self.identified_masses_table = self.identified_masses_table[self.identified_masses_table['mass shift'] >= 0]
        self.mass_shifts = self.identified_masses_table['mass shift'].values

# =============================================================================
#     def create_table_identified_masses(self, means, relative_abundances, column_name, bin_size_identified_masses):
#         if self.identified_masses_table.empty:
#             self.identified_masses_table['masses '+column_name] = means
#             self.identified_masses_table['rel. abundances '+column_name] = relative_abundances     
#         else:
#             self.identified_masses_table['masses '+column_name] = np.nan
#             self.identified_masses_table['rel. abundances '+column_name] = np.nan
#             for ix in range(len(means)):
#                 mass = means[ix]
#                 abundance = relative_abundances[ix]
#                 row_index = self.identified_masses_table[(self.identified_masses_table['average mass']<=mass+bin_size_identified_masses) & 
#                                                          (self.identified_masses_table['average mass']>=mass-bin_size_identified_masses)].index.values
#                 if len(row_index) > 0:
#                     self.identified_masses_table.loc[row_index[0]]['masses '+column_name] = mass
#                     self.identified_masses_table.loc[row_index[0]]['rel. abundances '+column_name] = abundance
#                 else:
#                     self.identified_masses_table = self.identified_masses_table.append({'masses '+column_name: mass,
#                                                                                         'rel. abundances '+column_name: abundance}, ignore_index=True)
#         
#         self.identified_masses_table['average mass'] = self.identified_masses_table.filter(regex='masses').mean(axis=1).values
#         self.identified_masses_table['mass std'] = self.identified_masses_table.filter(regex='masses').std(axis=1).values
#         self.identified_masses_table.sort_values(by='average mass', inplace=True)
# =============================================================================

# =============================================================================
#     def add_mass_shifts(self, unmodified_species_mass):
#         if unmodified_species_mass == 0:
#             self.identified_masses_table['mass shift'] = self.identified_masses_table['average mass'] - self.identified_masses_table['average mass'].iloc[0] 
#         else:
#             self.identified_masses_table['mass shift'] = self.identified_masses_table['average mass'] - unmodified_species_mass  
#         
#         self.identified_masses_table = self.identified_masses_table[self.identified_masses_table['mass shift'] >= 0]
#         self.mass_shifts = self.identified_masses_table['mass shift'].values
# =============================================================================
        
    def save_table_identified_masses(self, path_name):
        self.identified_masses_table = self.identified_masses_table.sort_index(axis=1)
        self.identified_masses_table.to_csv(path_name+'identified_masses_table.csv', sep=',', index=False)    


    def save_table_ptm_patterns(self, path_name):
        self.ptm_patterns_table = self.ptm_patterns_table.sort_index(axis=1)
        self.ptm_patterns_table.to_csv(path_name+'ptm_patterns_table.csv', sep=',', index=False) 
        
        
    # def determine_ptm_patterns(self, modifications, maximal_mass_error_array):
    #     self.ptm_patterns_table = self.__create_ptm_pattern_table(modifications)
    #     lp_model = LinearProgram()
    #     lp_model.set_ptm_mass_shifts(modifications.modification_masses)
    #     lp_model.set_upper_bounds(modifications.upper_bounds)
    #     minimal_mass_shift = min(np.abs(modifications.modification_masses))

    #     for ix in range(len(self.mass_shifts)):
    #         mass_shift = self.mass_shifts[ix]
    #         maximal_mass_error = 2*maximal_mass_error_array[ix]

    #         if mass_shift >= minimal_mass_shift:
    #             mass_error = 0.0     
    #             lp_model.reset_mass_error()
    #             lp_model.set_observed_mass_shift(mass_shift) 
    #             while abs(mass_error) < maximal_mass_error:            
    #                 lp_model.solve_linear_program_for_objective_with_absolute_value()
    #                 mass_error = lp_model.get_objective_value()
    #                 lp_model.set_minimal_mass_error(abs(mass_error))
    #                 ptm_pattern = lp_model.get_x_values()
    #                 number_ptm_types = np.array(ptm_pattern).sum()
    #                 if abs(mass_error) < maximal_mass_error:
    #                     row_entry = [mass_shift, mass_error, ptm_pattern, number_ptm_types]
    #                     row_entry_as_series = pd.Series(row_entry, index = self.ptm_patterns_table.columns)
    #                     self.ptm_patterns_table = self.ptm_patterns_table.append(row_entry_as_series, ignore_index=True)

    #         utils.progress(ix+1, len(self.mass_shifts))
    
    
    def determine_ptm_patterns(self, modifications, maximal_mass_error_array):
        self.ptm_patterns_table = self.__create_ptm_pattern_table(modifications)
        lp_model = LinearProgramCVXOPT(np.array(modifications.modification_masses), np.array(modifications.upper_bounds))
        minimal_mass_shift = min(np.abs(modifications.modification_masses))

        for ix in range(len(self.mass_shifts)):
            mass_shift = self.mass_shifts[ix]
            maximal_mass_error = maximal_mass_error_array[ix]

            if mass_shift >= minimal_mass_shift:
                lp_model.set_observed_mass_shift(mass_shift)
                lp_model.set_max_mass_error(maximal_mass_error)
                row_entries = []
                min_number_ptms = 0
                count_laps = 0 
                while count_laps < self.laps_run_lp:         
                    status, solution_min_ptm = lp_model.solve_lp_min_ptms(min_number_ptms)
                    if solution_min_ptm:
                        count_laps += 1
                        number_ptms = int(sum(solution_min_ptm))
                        min_number_ptms = number_ptms+1
                        ptm_pattern = tuple(map(int, solution_min_ptm))
                        error = lp_model.get_error(solution_min_ptm)
                        row_entries.append([mass_shift, error, ptm_pattern, number_ptms])
                        min_error = 0
                        multiplier = 1
                        while count_laps < self.laps_run_lp and min_error <= maximal_mass_error:
                            status, solution_min_error = lp_model.solve_lp_min_error(number_ptms, min_error)
                            if solution_min_error:
                                ptm_pattern = tuple(map(int, solution_min_error[:-1]))
                                is_solution_in_list = [True for prev_sol in row_entries if np.array_equal(np.array(prev_sol[2]), np.array(ptm_pattern))]
                                if not is_solution_in_list:
                                    count_laps += 1
                                    error = lp_model.get_error(solution_min_error[:-1])
                                    number_ptms = int(sum(solution_min_error[:-1]))
                                    row_entries.append([mass_shift, error, ptm_pattern, number_ptms])
                                    min_error = solution_min_error[-1] + 1e-3
                                else:
                                    min_error = solution_min_error[-1] + 1e-3*multiplier
                                    multiplier += 1
                            else:        
                                break
                    else:
                        break
                        
                row_entries_as_df = pd.DataFrame(row_entries, columns=self.ptm_patterns_table.columns)
                row_entries_as_df.sort_values(by=['amount of PTMs', 'mass error (Da)'], inplace=True)
                self.ptm_patterns_table = self.ptm_patterns_table.append(row_entries_as_df, ignore_index=True)

            utils.progress(ix+1, len(self.mass_shifts))
        print('\n')


    def estimate_maximal_mass_error(self, mass_error):
        maximal_mass_error = np.repeat(mass_error, len(self.mass_shifts))
        #mass_mean = self.identified_masses_table['average mass'].values
        #mass_std = self.identified_masses_table['mass std'].values
        #maximal_mass_error = [mass_mean[i]*mass_error*1.0e-6 if np.isnan(mass_std[i]) else mass_std[i] for i in range(len(mass_std))]
        #maximal_mass_error = [mass_mean[i]*mass_error*1.0e-6+mass_std[0] if np.isnan(mass_std[i]) else mass_std[0]+mass_std[i] for i in range(len(mass_std))]
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


    def bin_peaks(self, bin_size_mass_shifts):
        peak_mass = self.identified_masses_table.filter(regex='masses').columns
        peak_abundance = self.identified_masses_table.filter(regex='rel. abundances ').columns
        for sample in range(len(peak_mass)):
            not_empty_rows_ix = self.identified_masses_table[~pd.isnull(self.identified_masses_table[peak_mass[sample]])].index
            for row_index in not_empty_rows_ix:
                row_index_updated = self.__count_non_empty_cells(row_index, bin_size_mass_shifts, peak_mass)
                self.identified_masses_table.at[row_index_updated, peak_mass[sample]] = self.identified_masses_table.loc[row_index, peak_mass[sample]]
                self.identified_masses_table.at[row_index_updated, peak_abundance[sample]] = self.identified_masses_table.loc[row_index, peak_abundance[sample]]
                if row_index != row_index_updated:
                    self.identified_masses_table.at[row_index, peak_mass[sample]] = np.nan
                    self.identified_masses_table.at[row_index, peak_abundance[sample]] = np.nan

        self.identified_masses_table['average mass'] = self.identified_masses_table.filter(regex='masses').mean(axis=1).values
        self.identified_masses_table['mass std'] = self.identified_masses_table.filter(regex='masses').std(axis=1).values
        
        
        no_empty_rows_df = self.identified_masses_table.dropna(axis=0, subset=peak_mass, how='all', inplace=False)
        #print(no_empty_rows_df['average mass'][1:])
        distance_between_indices = np.abs(no_empty_rows_df.index[1:]-no_empty_rows_df.index[:-1])
        print(distance_between_indices)
        #if min(distance_between_indices) <= bin_size_mass_shifts:
        #    self.bin_peaks(bin_size_mass_shifts)
        #else:
        self.identified_masses_table.dropna(axis=0, subset=peak_mass, how='all', inplace=True)
        self.identified_masses_table.sort_values(by='average mass', inplace=True)
                
    def __count_non_empty_cells(self, row_index, bin_size, columns):
        current_row_count = self.identified_masses_table.loc[row_index, columns].count().sum()
        non_empty_cells_per_row = {}
        distance = 1
        while distance <= bin_size:
            prev_row_count = self.identified_masses_table.loc[row_index-distance, columns].count().sum()
            next_row_count = self.identified_masses_table.loc[row_index+distance, columns].count().sum()
            if prev_row_count > 0 and prev_row_count >= current_row_count:
                non_empty_cells_per_row[-distance] = abs(self.identified_masses_table.loc[row_index, 'average mass'] - self.identified_masses_table.loc[row_index-distance, 'average mass'])
            if next_row_count > 0 and next_row_count >= current_row_count:
                non_empty_cells_per_row[distance] = abs(self.identified_masses_table.loc[row_index, 'average mass'] - self.identified_masses_table.loc[row_index+distance, 'average mass'])
            if not non_empty_cells_per_row:
                distance += 1
            else: 
                break

        if not non_empty_cells_per_row:
            return row_index
        else:
            
            distance_to_row_max_count = min(non_empty_cells_per_row, key=non_empty_cells_per_row.get)

            row_index_updated = row_index+distance_to_row_max_count
            return row_index_updated

    # def groupby_ptm_patterns(self):  
    #     no_ptm_pattern_identified_df = self.identified_masses_table[self.identified_masses_table['PTM pattern ('+self.ptm_column_name+')'].isna()]
    #     unmodified_species_df = self.identified_masses_table[(self.identified_masses_table['average mass']<=config.unmodified_species_mass_init+config.unmodified_species_mass_tol) & 
    #                                                          (self.identified_masses_table['average mass']>=config.unmodified_species_mass_init-config.unmodified_species_mass_tol)]       
    #     no_ptm_pattern_identified_df = no_ptm_pattern_identified_df[~no_ptm_pattern_identified_df['average mass'].isin(unmodified_species_df['average mass'])]
    #     unmodified_species_df = unmodified_species_df.groupby('PTM pattern ('+self.ptm_column_name+')', dropna=False).max()
    #     unmodified_species_df.reset_index(inplace=True)
    #     grouped_ptm_patterns_df = self.identified_masses_table.groupby('PTM pattern ('+self.ptm_column_name+')', dropna=True).max()
    #     grouped_ptm_patterns_df.reset_index(inplace=True)
    #     self.identified_masses_table = pd.concat([unmodified_species_df, no_ptm_pattern_identified_df, grouped_ptm_patterns_df], join='inner')
    #     self.identified_masses_table['average mass'] = self.identified_masses_table.filter(regex='masses').mean(axis=1).values
    #     self.identified_masses_table['mass std'] = self.identified_masses_table.filter(regex='masses').std(axis=1).values
    #     self.identified_masses_table.sort_values(by='average mass', inplace=True)

