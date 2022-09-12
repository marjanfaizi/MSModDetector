#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 6 2021

@author: Marjan Faizi
"""

import pandas as pd
import numpy as np
from linear_program_cvxopt import LinearProgramCVXOPT
import utils


class MassShifts(object): 
    """
    This class takes the results from fitting gaussian distributions to the observed isotope distributuons and calculates the mass shifts.
    For calculating the mass shifts the mass of an unmodified species has to be determined first.
    For a given mass shift the linear program determines global PTM patterns/combinations and reports the optimal combination with respect to the objective function.
    """
    
    def __init__(self, mass_start_range,mass_end_range):
        mass_index = np.arange(int(mass_start_range), int(mass_end_range+20))
        self.identified_masses_df = pd.DataFrame(index=mass_index)
        self.mass_shifts = []
        self.mass_col_name = "masses "
        self.intensity_col_name = "raw intensities "
        self.abundance_col_name = "rel. abundances "
        self.avg_mass_col_name = "average mass"
        self.pvalue_col_name = "pvalue "
        self.chi_score_col_name = "chi_score "


    def add_identified_masses_to_df(self, fitting_results, column_name, rescaling_factor=None):
        means = fitting_results["mean"].values
        if rescaling_factor: amplitudes = fitting_results["amplitude"].values*rescaling_factor
        else: amplitudes = fitting_results["amplitude"].values
        relative_abundances = fitting_results["relative_abundance"].values
        self.identified_masses_df.loc[means.round().astype(int), self.mass_col_name+column_name] = means
        self.identified_masses_df.loc[means.round().astype(int), self.intensity_col_name+column_name] = amplitudes
        self.identified_masses_df.loc[means.round().astype(int), self.abundance_col_name+column_name] = relative_abundances
    
        pvalue = fitting_results["pvalue"].values
        chi_score = fitting_results["chi_score"].values
        self.identified_masses_df.loc[means.round().astype(int), self.pvalue_col_name+column_name] = pvalue
        self.identified_masses_df.loc[means.round().astype(int), self.chi_score_col_name+column_name] = chi_score


    def bin_peaks(self, max_bin_size):
        max_distances_between_bins = self.identified_masses_df.filter(regex="masses ").max(axis=1).values[1:] - \
                                     self.identified_masses_df.filter(regex="masses ").min(axis=1).values[:-1]  
        
        while (max_distances_between_bins <= max_bin_size).any():
            distance_matrix = self.__create_distance_matrix(self.identified_masses_df[self.avg_mass_col_name].values)
            indices_to_combine = np.unravel_index(np.nanargmin(distance_matrix, axis=None), distance_matrix.shape)
            self.identified_masses_df.iloc[indices_to_combine[0]] = self.identified_masses_df.iloc[list(indices_to_combine)].max()
            self.identified_masses_df.iloc[indices_to_combine[1]] = np.nan 
            self.calculate_avg_mass()
            max_distances_between_bins = self.identified_masses_df.filter(regex="masses ").max(axis=1).values[1:] - \
                                         self.identified_masses_df.filter(regex="masses ").min(axis=1).values[:-1]  


    def calculate_avg_mass(self):
        self.identified_masses_df[self.avg_mass_col_name] = self.identified_masses_df.filter(regex=self.mass_col_name).mean(axis=1).values
        self.identified_masses_df.dropna(axis=0, how='all', inplace=True)
        self.identified_masses_df.reset_index(drop=True, inplace=True)


    def __create_distance_matrix(self, array):
        m, n = np.meshgrid(array, array)
        distance_matrix = abs(m-n)
        arbitrary_high_number = 100
        np.fill_diagonal(distance_matrix, arbitrary_high_number)
        return distance_matrix


    def add_mass_shifts(self, unmodified_species_mass):
        self.identified_masses_df['mass shift'] = self.identified_masses_df[self.avg_mass_col_name] - unmodified_species_mass 
        self.mass_shifts = self.identified_masses_df['mass shift'].values


    def save_tables(self, output_path_name):
        regex_score_cols =  "mass shift"+"|"+self.pvalue_col_name+"|"+self.chi_score_col_name
        regex_mass_shift_cols = "mass shift"+"|"+"PTM pattern"+"|"+self.avg_mass_col_name+ "|"+\
                                self.mass_col_name+"|"+self.intensity_col_name+"|"+self.abundance_col_name

        masses_df = self.identified_masses_df.filter(regex=regex_mass_shift_cols)
        score_df = self.identified_masses_df.filter(regex=regex_score_cols)
                
        masses_df.sort_index(axis=1).to_csv(output_path_name+"mass_shifts.csv", sep=',', index=False) 
        score_df.to_csv(output_path_name+"scores.csv", sep=',', index=False)


    def determine_ptm_patterns(self, modifications, mass_tolerance, objective_fun, laps_run_lp, msg_progress=True):
        self.ptm_patterns_df = pd.DataFrame(columns=["mass shift", "mass error (Da)", "PTM pattern", "amount of PTMs"]) 
        lp_model = LinearProgramCVXOPT(np.array(modifications.ptm_masses), np.array(modifications.upper_bounds))
        lp_model.set_max_mass_error(mass_tolerance)

        progress_bar_count = 0
        for mass_shift in self.mass_shifts:
            if mass_shift-np.array(modifications.ptm_masses).min() > -mass_tolerance:
                lp_model.set_observed_mass_shift(mass_shift)         
                row_entries = []
                count_laps = 0 
    
                min_number_ptms = 0
                min_error = 0
                min_error_and_ptms = 0
                multiplier = 1

                if objective_fun == "min_ptm":    
                    while count_laps < laps_run_lp:
                        status, solution_min_ptm = lp_model.solve_lp_ptms(min_number_ptms, optimization_type="min")
                        if solution_min_ptm:
                            count_laps += 1
                            number_ptms = int(sum(solution_min_ptm))
                            ptm_pattern = self.array_to_ptm_annotation(list(solution_min_ptm), modifications.ptm_ids)
                            error = lp_model.get_error(solution_min_ptm)
                            row_entries.append([mass_shift, error, ptm_pattern, number_ptms])
                            min_error = 0
                            multiplier = 1
                            while count_laps < laps_run_lp and min_error <= mass_tolerance:
                                status, solution_min_error = lp_model.solve_lp_min_error(min_error, number_ptms)
                                if solution_min_error:
                                    ptm_pattern = self.array_to_ptm_annotation(list(solution_min_error[:-1]), modifications.ptm_ids)
                                    is_solution_in_list = [True for prev_sol in row_entries if np.array_equal(np.array(prev_sol[2]), np.array(ptm_pattern))]
                                    if not is_solution_in_list:
                                        count_laps += 1
                                        number_ptms = int(sum(solution_min_error[:-1]))
                                        error = solution_min_error[-1]
                                        row_entries.append([mass_shift, error, ptm_pattern, number_ptms])
                                        min_error = error + 1e-4
                                    else:
                                        min_error = error + 1e-4*multiplier
                                        multiplier += 1
                                else:        
                                    break
                            min_number_ptms = number_ptms+1
                        else:
                            break
                
                if objective_fun == "min_err": 
                    while count_laps < laps_run_lp and min_error <= mass_tolerance:
                        status, solution_min_error = lp_model.solve_lp_min_error(min_error)
                        if solution_min_error:
                            ptm_pattern = self.array_to_ptm_annotation(list(solution_min_error[:-1]), modifications.ptm_ids)
                            is_solution_in_list = [True for prev_sol in row_entries if np.array_equal(np.array(prev_sol[2]), np.array(ptm_pattern))]
                            if not is_solution_in_list:
                                count_laps += 1
                                number_ptms = int(sum(solution_min_error[:-1]))
                                error = solution_min_error[-1]
                                row_entries.append([mass_shift, error, ptm_pattern, number_ptms])
                                min_error = error + 1e-4
                            else:
                                min_error = error + 1e-4*multiplier
                                multiplier += 1
                        else:        
                            break
    
                if objective_fun == "min_both":
                    status, solution_max_ptm = lp_model.solve_lp_ptms(min_number_ptms, optimization_type="max")
                    if solution_max_ptm:
                        max_number_ptms = int(sum(solution_max_ptm))
                        while count_laps < laps_run_lp and min_error <= mass_tolerance:
                            status, solution_min_both = lp_model.solve_lp_min_both(min_error_and_ptms, max_number_ptms)
                            if solution_min_both:
                                ptm_pattern = self.array_to_ptm_annotation(list(solution_min_both[:-1]), modifications.ptm_ids)
                                is_solution_in_list = [True for prev_sol in row_entries if np.array_equal(np.array(prev_sol[2]), np.array(ptm_pattern))]
                                if not is_solution_in_list:
                                    count_laps += 1
                                    error = solution_min_both[-1]
                                    number_ptms = int(sum(solution_min_both[:-1]))
                                    row_entries.append([mass_shift, error, ptm_pattern, number_ptms])
                                    min_error_and_ptms = error/mass_tolerance + number_ptms/max_number_ptms + 1e-4
                                else:
                                    min_error_and_ptms = error/mass_tolerance + number_ptms/max_number_ptms + 1e-4*multiplier
                                    multiplier += 1
                            else:        
                                break
                    else:
                        break

                row_entries_as_df = pd.DataFrame(row_entries, columns=self.ptm_patterns_df.columns)
                if objective_fun == "min_ptm":
                    row_entries_as_df.sort_values(by=["amount of PTMs", "mass error (Da)"], inplace=True)
                self.ptm_patterns_df = self.ptm_patterns_df.append(row_entries_as_df, ignore_index=True)

            if msg_progress:
                progress_bar_count += 1
                utils.progress(progress_bar_count, len(self.mass_shifts))



    def array_to_ptm_annotation(self, ptm_array, ptm_acronyms):
        ptm_pattern = ""
        for ix, ptm_count in enumerate(ptm_array):
            if ptm_count > 0:
                ptm_pattern = ptm_pattern+str(int(ptm_count))+"["+ptm_acronyms[ix]+"]"
        
        return ptm_pattern
        
    
    def add_ptm_patterns_to_table(self):
        # Delete PTM pattern column if it already exists
        self.identified_masses_df.drop("PTM pattern", inplace=True, axis=1, errors="ignore")
        best_ptm_patterns = self.ptm_patterns_df.groupby("mass shift").head(1)

        self.identified_masses_df = pd.merge(best_ptm_patterns[["mass shift", "PTM pattern"]], 
                                             self.identified_masses_df, how="outer", left_on=["mass shift"], 
                                             right_on=["mass shift"])
        self.identified_masses_df.sort_values(by=["mass shift"], inplace=True, ignore_index=True)
        self.identified_masses_df["PTM pattern"].fillna("0", inplace=True)



# =============================================================================
#     def align_spetra(self):
#         unmodified_species_masses = self.identified_masses_df.filter(regex=self.mass_col_name).apply(utils.determine_unmodified_species_mass, raw=True, args=(config.unmodified_species_mass_init, config.unmodified_species_mass_tol)).values
#         unmodified_species_average = unmodified_species_masses.mean()
#         calibration_number = unmodified_species_average - unmodified_species_masses
#         aligned_masses_df = self.identified_masses_df.copy()
#         columns_to_correct = self.identified_masses_df.filter(regex=self.mass_col_name).columns
#         aligned_masses_df[columns_to_correct] = aligned_masses_df[columns_to_correct] + calibration_number
#         sample_names = [column[len(self.mass_col_name):] for column in columns_to_correct]
#         for col_name in sample_names:
#             indices = aligned_masses_df[self.mass_col_name+col_name].dropna().round().astype(int).values
#             masses = aligned_masses_df[self.mass_col_name+col_name].dropna().values
#             intensities = aligned_masses_df[self.intensity_col_name+col_name].dropna().values
#             abundances = aligned_masses_df[self.abundance_col_name+col_name].dropna().values
#             self.identified_masses_df[[self.mass_col_name+col_name, self.abundance_col_name+col_name, self.intensity_col_name+col_name]] = np.nan
#             self.identified_masses_df.loc[indices, self.mass_col_name+col_name] = masses
#             self.identified_masses_df.loc[indices, self.intensity_col_name+col_name] = intensities
#             self.identified_masses_df.loc[indices, self.abundance_col_name+col_name] = abundances
# 
#         self.identified_masses_df.dropna(axis=0, how='all', inplace=True)
#         self.identified_masses_df.reset_index(drop=True, inplace=True)
#         
#         all_samples = [name[len(self.mass_col_name):] for name in columns_to_correct]
#         calibration_number_dict = dict(zip(all_samples, calibration_number))
#         return calibration_number_dict
# =============================================================================


