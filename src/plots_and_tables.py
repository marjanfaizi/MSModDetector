#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 2021

@author: Marjan Faizi
"""

import numpy as np
from fit_gaussian_model import multi_gaussian
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class PlotsAndTables(object):
  
    sns.set_context("paper", font_scale=1.2)  
        
        
    def __init__(self):
        self.identified_masses_table = pd.DataFrame(columns=['mass mean', 'mass std']) 
   
        
    def get_table_identified_masses(self):  
        return self.identified_masses_table
    
        
    def save_table_identified_masses(self, path_name):
        self.identified_masses_table = self.identified_masses_table.sort_index(axis=1)
        self.identified_masses_table.to_csv(path_name+'identified_masses_table.csv', sep=',')    
        return self.identified_masses_table
    
    
    def plot_mass_shifts(self, spectrum, peaks, mass_shifts, mass_range_start, mass_range_end, stddev, title):
        self.__plot_spectra(spectrum, peaks, mass_range_start, mass_range_end, mass_shifts.mean, mass_shifts.amplitude, stddev, title)
        self.__add_labels(mass_shifts.mass_shifts, mass_shifts.mean, mass_shifts.amplitude)

    def plot_masses(self, spectrum, peaks, mass_range_start, mass_range_end, mean, amplitude, stddev, title):
        self.__plot_spectra(spectrum, peaks, mass_range_start, mass_range_end, mean, amplitude, stddev, title)
        self.__add_labels(mean, mean, amplitude)


    def __plot_spectra(self, spectrum, peaks, mass_range_start, mass_range_end, mean, amplitude, stddev, title):
        x = np.arange(mass_range_start, mass_range_end)
        y = multi_gaussian(x, amplitude, mean, stddev)
        masses = spectrum[:,0]
        intensities = spectrum[:,1]
        plt.figure(figsize=(16,4))
        plt.plot(masses, intensities, color='gray')
        plt.plot(peaks[:,0], peaks[:,1], '.b')
        plt.plot(mean, amplitude, '.r')
        plt.plot(x, y, color='firebrick')
        plt.xlim((mass_range_start, mass_range_end))
        plt.ylim((-1, peaks[:,1].max()*1.2))
        plt.title('Sample: '+title)
        plt.xlabel('mass (Da)')
        plt.ylabel('relative intensity (%)')
        plt.legend(['I2MS data', 'Mass shifts', 'Fitted normal distributions'])
        plt.tight_layout()
        sns.despine()


    def __add_labels(self, label, mean, amplitude):   
        for ix in range(len(label)):
            label_str = "{:.2f}".format(label[ix])
            plt.annotate(label_str, (mean[ix], amplitude[ix]), textcoords="offset points", xytext=(5,25), 
                         ha='center', fontsize='small', rotation=45)
     
    
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


    def create_table_identified_masses(self, fitting_results, column_name, bin_size_identified_masses):
        if len(fitting_results) == 0:
            avg_masses = np.nan
        else:
            avg_masses = list(fitting_results.keys())
        
        if self.identified_masses_table.empty:
            self.identified_masses_table[column_name] = avg_masses
        else:
            self.identified_masses_table[column_name] = np.nan
            for mass in avg_masses:
                row_index = self.identified_masses_table[(self.identified_masses_table['mass mean']<=mass+bin_size_identified_masses) & 
                                                         (self.identified_masses_table['mass mean']>=mass-bin_size_identified_masses)].index.values
                if len(row_index) > 0:
                    self.identified_masses_table.loc[row_index[0]][column_name] = mass
                else:
                    self.identified_masses_table = self.identified_masses_table.append({column_name: mass}, ignore_index=True)
        
        self.identified_masses_table['mass mean'] = self.identified_masses_table.iloc[:,2:].mean(axis=1).values
        self.identified_masses_table['mass std'] = self.identified_masses_table.iloc[:,2:].std(axis=1).values
        self.identified_masses_table.sort_values(by='mass mean', inplace=True)


    def add_mass_shifts(self):
        self.identified_masses_table['mass shift'] = self.identified_masses_table['mass mean'] - self.identified_masses_table.loc[0]['mass mean']
  

    def add_ptm_patterns(self, ptm_patterns, output_df, mod):
        ptm_column_name_separator = ', '
        ptm_column_name = ptm_column_name_separator.join(mod.acronyms)
        column_name = 'PTM pattern ('+ptm_column_name+')'
        self.identified_masses_table['best '+column_name] = np.nan
        
        for ix, row in output_df.iterrows():
            row_index = self.identified_masses_table[self.identified_masses_table['mass shift']==row['mass shift']].index.values[0] 
            self.identified_masses_table.loc[row_index,'best '+column_name] = str(row[column_name])










