#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2021

@author: Marjan Faizi
"""

import glob
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
import utils

# Should be changed for different configurations
import myconfig as config

file_names = [file for file in glob.glob(config.path+config.file_name_ending)] 
 
if __name__ == '__main__':
    
    print('\n'+'-'*80)     
    print('\nLoad files...')
    
    if not file_names:
        print('\nFile does not exist.\n')
        sys.exit()
    
    mod = Modifications(config.modfication_file_name)
    mod.get_modification_masses()

    mass_shifts = MassShifts()
     
    output_fig = plt.figure(figsize=(14,7))
    gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
    axes = gs.subplots(sharex=True, sharey=True)
    ylim_max = 0
    
    print('\nStart detecting mass shifts...')
    
    stdout_text = []
    #sample_names_list = []
        
    progress_bar_mass_shifts = 0

    for file_name in file_names:
                
        text_extract = re.search(config.regex_extract_output_name, file_name)
        if not text_extract:
            print('\nRegular expression could not be found in file names.\n')
            sys.exit()
        else: 
            sample_name = text_extract.group(1).lower()
            #sample_names_list.append(sample_name)

        data = MassSpecData(file_name)
        data.set_mass_error(config.mass_error)
        data.set_max_mass_shift(config.max_mass_shift)
        data.set_search_window_mass_range(config.start_mass_range)        
  
        max_intensity = data.intensities.max()
        rescaling_factor = max_intensity/100.0
        data.convert_to_relative_intensities(max_intensity)
      
        all_peaks = data.picking_peaks()
        trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
        trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
  
        color_of_sample = [value[0] for key, value in config.color_palette.items() if key in '\_'+sample_name][0]
        order_in_plot = [value[1] for key, value in config.color_palette.items() if key in '\_'+sample_name][0]
        axes[order_in_plot].plot(data.masses, data.intensities*rescaling_factor, label=sample_name, color=color_of_sample)

        if trimmed_peaks_in_search_window.size:
            ### TODO: is there another way to calculate th s/n ratio?
            sn_threshold = trimmed_peaks_in_search_window[:,1].std()/2
            ### TODO: remove later, just to show sn threshold
            axes[order_in_plot].axhline(y=sn_threshold*rescaling_factor, c='r', lw=0.3)
            
            peaks_above_sn = data.picking_peaks(min_peak_height=sn_threshold)
            trimmed_peaks_above_sn  = data.remove_adjacent_peaks(peaks_above_sn, config.distance_threshold_adjacent_peaks)  
            trimmed_peaks_above_sn_in_search_window = data.determine_search_window(trimmed_peaks_above_sn)
        
            if trimmed_peaks_above_sn_in_search_window.size:  
                # 1. ASSUMPTION: The isotopic distribution follows a normal distribution.
                # 2. ASSUMPTION: The standard deviation does not change when modifications are included to the protein mass. 
                gaussian_model = GaussianModel(sample_name)
                gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass_init)
                gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, trimmed_peaks_above_sn_in_search_window)

                #axes[order_in_plot].plot(gaussian_model.fitting_results['means'], gaussian_model.fitting_results['amplitudes']*rescaling_factor, 'r.')

                gaussian_model.filter_fitting_results(config.pvalue_threshold)
                gaussian_model.refit_amplitudes(trimmed_peaks_in_search_window, sn_threshold)
                gaussian_model.calculate_relative_abundaces(data.search_window_start_mass, data.search_window_end_mass)
        
                mass_shifts.create_table_identified_masses(gaussian_model.fitting_results['means'], gaussian_model.fitting_results['relative_abundances'], sample_name)

                if not gaussian_model.fitting_results.empty:
                    x_gauss_func = np.arange(data.search_window_start_mass, data.search_window_end_mass)
                    y_gauss_func = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results['amplitudes'], gaussian_model.fitting_results['means'], 
                                                        gaussian_model.fitting_results['stddevs'])
                    axes[order_in_plot].plot(x_gauss_func, y_gauss_func*rescaling_factor, color='0.3')
                    axes[order_in_plot].plot(gaussian_model.fitting_results['means'], gaussian_model.fitting_results['amplitudes']*rescaling_factor, '.', color='0.3')
                    
                    #utils.add_labels(gaussian_model.fitting_results['means'], gaussian_model.fitting_results['amplitudes']*rescaling_factor, gaussian_model.fitting_results['p-values'])
                    
                    if ylim_max < trimmed_peaks_in_search_window[:,1].max()*rescaling_factor:
                        ylim_max = trimmed_peaks_in_search_window[:,1].max()*rescaling_factor
                    
                else:
                    stdout_text.append('No masses detected for the following condition: ' + sample_name)
            else:
                stdout_text.append('No peaks above the SN threshold could be detected within the search window for the following condition: ' + sample_name)

        else:
            stdout_text.append('No peaks could be detected within the search window for the following condition: ' + sample_name)

        progress_bar_mass_shifts += 1        
        utils.progress(progress_bar_mass_shifts, len(file_names))
    
    print('\n')
    seperator_stdout_text = '\n'
    print(seperator_stdout_text.join(stdout_text))
    
    if mass_shifts.identified_masses_table.empty:
        print('\nNo masses detected.')
    else:
        mass_shifts.set_index()
        mass_shifts.bin_peaks(config.bin_size_mass_shifts)
        if config.calculate_mass_shifts == True:
            mass_shifts.add_mass_shifts()
            mass_shifts.save_table_identified_masses('../output/')
        else:
            mass_shifts.save_table_identified_masses('../output/')
    
        if (config.calculate_mass_shifts == True) and (config.determine_ptm_patterns == True):
            maximal_mass_error = mass_shifts.estimate_maximal_mass_error(config.mass_error)
            
            print('\nSearching for PTM combinations:')
        
            mass_shifts.determine_ptm_patterns(mod, maximal_mass_error)        
            mass_shifts.add_ptm_patterns_to_table()
            mass_shifts.groupby_ptm_patterns()
            mass_shifts.save_table_identified_masses('../output/')
            mass_shifts.save_table_ptm_patterns('../output/')
            
            
    means = mass_shifts.identified_masses_table['average mass'].values
    for ax in axes:
        ax.set_xlim((data.search_window_start_mass-10, data.search_window_end_mass+10))
        ax.set_ylim((-10, ylim_max*1.1))
        ax.yaxis.grid()
        ax.legend(fontsize=11, loc='upper right')
        for m in means:
            ax.axvline(x=m, c='0.3', ls='--', lw=0.3, zorder=0)
        ax.label_outer()
    plt.xlabel('mass (Da)'); plt.ylabel('intensity')
    output_fig.tight_layout()
    plt.savefig('../output/identified_masses.pdf', dpi=800)
    plt.show()
    
    print('\n')
    print(80*'-'+'\n\n')

