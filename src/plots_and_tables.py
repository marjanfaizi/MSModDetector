#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 2021

@author: Marjan Faizi
"""

import numpy as np
import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class PlotsAndTables(object):
  
    sns.set_context("paper", font_scale=1.2)  
        
        
    def __init__(self):
        self.identified_masses_table = pd.DataFrame(columns=['mass mean', 'mass std'])     
    
    
    def plot_mass_shifts(self, masses, intensities, all_peaks, trimmed_peaks, mass_shifts, mass_range_start, mass_range_end, stddev, title):
        self.__plot_spectra(masses, intensities, all_peaks, trimmed_peaks, mass_range_start, mass_range_end, mass_shifts.mean, mass_shifts.amplitude, stddev, title)
        self.__add_labels(mass_shifts.mass_shifts, mass_shifts.mean, mass_shifts.amplitude)


    def plot_masses(self, masses, intensities, all_peaks, trimmed_peaks, mass_range_start, mass_range_end, mean, amplitude, stddev, title):
        self.__plot_spectra(masses, intensities, all_peaks, trimmed_peaks, mass_range_start, mass_range_end, mean, amplitude, stddev, title)
        self.__add_labels(mean, mean, amplitude)


    def __plot_spectra(self, masses, intensities, all_peaks, trimmed_peaks, mass_range_start, mass_range_end, mean, amplitude, stddev, title):
        x = np.arange(mass_range_start, mass_range_end)
        y = utils.multi_gaussian(x, amplitude, mean, stddev)
        plt.figure(figsize=(16,4))
        plt.plot(masses, intensities, color='gray')
        plt.plot(all_peaks[:,0], all_peaks[:,1], '.g')
        plt.plot(trimmed_peaks[:,0], trimmed_peaks[:,1], '.b')
        plt.plot(mean, amplitude, '.r')
        plt.plot(x, y, color='firebrick')
        plt.xlim((mass_range_start-10, mass_range_end+10))
        plt.ylim((-1, trimmed_peaks[:,1].max()*1.3))
        plt.title('Sample: '+title+'\n\n')
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


    def add_ptm_patterns(self, ptm_patterns, output_df, mod):
        ptm_column_name_separator = ', '
        ptm_column_name = ptm_column_name_separator.join(mod.acronyms)
        column_name = 'PTM pattern ('+ptm_column_name+')'
        self.identified_masses_table['best '+column_name] = np.nan
        
        for ix, row in output_df.iterrows():
            row_index = self.identified_masses_table[self.identified_masses_table['mass shift']==row['mass shift']].index.values[0] 
            self.identified_masses_table.loc[row_index,'best '+column_name] = str(row[column_name])

