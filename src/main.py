#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 2021

@author: Marjan Faizi
"""

import numpy as np
import glob
import re
from matplotlib.backends.backend_pdf import PdfPages

from i2ms_data import MassSpecData
from fit_gaussian_model import FitGaussianModel
from protein import Protein
from mass_shifts import MassShifts
from modifications import Modifications
from plots_and_tables import PlotsAndTables


##################################################################################################### 
############################################### INPUT ############################################### 
##################################################################################################### 
"""
file_names =  ['/Users/marjanfaizi/Documents/Postdoc/Data/TopDown/03_10_2021/20210310_bsd0560_SS_A375-wt_MEK1_FA_i2ms_01_Profile.mzml']
#file_names =  '/Users/marjanfaizi/Documents/Postdoc/Data/TopDown/03_10_2021/20210310_bsd0560_SS_A375-wt_MEK1_FA_i2ms_01_Cent.mzml'
protein_id = 'Q02750'
modfication_file_name = '/Users/marjanfaizi/Documents/Postdoc/Code/data/modifications_Q02750.csv'
max_mass_shift = 600.0 #Da
start_mass_range = 43200.0 #Da
unmodified_species_mass_init = 43310.0 #Da
unmodified_species_mass_tol = 5.0 #Da
regex_extract_output_name = 'wt_(.*)_Profile'
"""
#file_names =  ['/Users/marjanfaizi/Documents/Postdoc/Data/TopDown/05_04_2021/I2MS_20210504_bsd0560_p53_MCF7_7hr-sustained_MS1_Profile.txt']
#file_names =  ['/Users/marjanfaizi/Documents/Postdoc/Data/TopDown/05_04_2021/I2MS_20210504_bsd0560_p53_MCF7_2hr_MS1_Profile.txt']
#file_names =  ['/Users/marjanfaizi/Documents/Postdoc/Data/TopDown/06_22_2021/I2MS_20210622_bsd0560_p53_MCF7_nutlin_7hr_rep1_MS1_02_Profile.mzml']
path = '/Users/marjanfaizi/Documents/Postdoc/Data/TopDown/06_22_2021/'
file_names = [file for file in glob.glob(path+'*.mzml')]
regex_extract_output_name = 'MCF7_(.*)_Profile'
#protein_id = 'P04637'
protein_id = 'P01286'
modfication_file_name = '/Users/marjanfaizi/Documents/Postdoc/Code/data/modifications_P04637.csv'
max_mass_shift = 200.0 # 900.0 #Da
start_mass_range = 12400.0 # 43700.0 #Da
unmodified_species_mass_init = 12445.0 # 43770.0 #Da
unmodified_species_mass_tol = 5.0 #Da


fasta_name = '/Users/marjanfaizi/Documents/Postdoc/Code/data/'+protein_id+'.fasta'
output_path = '/Users/marjanfaizi/Documents/Postdoc/Code/output/'
output_plots_name = 'TDMS_07_03_2021.pdf'
mass_error = 5.0 #ppm
pvalue_threshold = 0.8#np.arange(0.9, 0.999, 0.01)
distance_threshold_adjacent_peaks = 0.6
bin_size_identified_masses = 2.0 #Da
calculate_mass_shifts = True
determine_ptm_patterns = False
# TODO: bessere beschriftgun
top_results = 1
##################################################################################################### 


#####################################################################################################
######################################### HELPER FUNCTIONS  #########################################
#####################################################################################################
def adaptive_window_sizes(distribution):
    intensities = distribution[:,1] 
    most_abundant_masses_25percent = distribution[intensities > intensities.max()*0.75, 0]
    most_abundant_masses_95percent = distribution[intensities > intensities.max()*0.05, 0]
    lower_window_size = round((most_abundant_masses_25percent[-1]-most_abundant_masses_25percent[0])/2)
    upper_window_size = np.ceil((most_abundant_masses_95percent[-1]-most_abundant_masses_95percent[0])/2)
    window_sizes = np.arange(lower_window_size, upper_window_size, 1)
    return window_sizes
#####################################################################################################


if __name__ == '__main__':
    
    print('\n'+'-'*80)  
    
    protein = Protein(fasta_name)
    theoretical_distribution = protein.get_theoretical_isotope_distribution(fasta_name)
    average_protein_mass = protein.get_average_mass()

    mod = Modifications(modfication_file_name)
    mod.get_modification_masses()
 
    plots_tables_obj = PlotsAndTables()
    
    window_sizes = adaptive_window_sizes(theoretical_distribution)
 
    pp = PdfPages(output_path+output_plots_name)
    
    file_count = 1
    for file_name in file_names:
        
        print('\n'+'#'*20) 
        print(file_count, '/', len(file_names)) 
        print('#'*20) 
        file_count+=1
        
        sample_name = re.search(regex_extract_output_name, file_name).group(1).lower()
    
        data = MassSpecData(file_name)
        data.convert_to_relative_intensities()
        data.set_mass_error(mass_error)
        data.set_max_mass_shift(max_mass_shift)
        data.set_search_window_mass_range(start_mass_range)  
        
        all_peaks = data.get_peaks()
        all_peaks_trimmed  = data.remove_adjacent_peaks(all_peaks, distance_threshold_adjacent_peaks)   
        all_peaks_in_search_window = data.determine_search_window(all_peaks_trimmed)
        # TODO: is there another way to calculate th s/n ratio?
        sn_threshold = all_peaks_in_search_window[:,1].mean()/all_peaks_in_search_window[:,1].std()   
        
        peaks_above_sn = data.get_peaks(min_peak_height=sn_threshold)
        peaks_above_sn_trimmed  = data.remove_adjacent_peaks(peaks_above_sn, distance_threshold_adjacent_peaks)  
        peaks_above_sn_in_search_window = data.determine_search_window(peaks_above_sn_trimmed)
    
        print('\nMass shifts are calculated.') 
    
        # 1. ASSUMPTION: The isotopic distribution follows a normal distribution.
        # 2. ASSUMPTION: The standard deviation does not change when modifications are included to the protein mass. 
        gaussian_model = FitGaussianModel()
        optimized_param_theorerical_distribution = gaussian_model.fit_gaussian(theoretical_distribution)
        gaussian_model.set_standard_deviation(optimized_param_theorerical_distribution[2])    
        fitting_results = gaussian_model.fit_gaussian_to_single_peaks(all_peaks_trimmed, peaks_above_sn_in_search_window, window_sizes)
        
        best_fitting_results = gaussian_model.filter_fitting_results(fitting_results, pvalue_threshold)
        """
        # optimize window size for fitting the gaussian model
        best_score = -10.0
        best_fitting_results = {}
        for pvalue in pvalue_threshold:
            fitting_results_reduced = gaussian_model.filter_fitting_results(fitting_results, pvalue)
            if len(fitting_results_reduced) == 0:
                continue
            adjusted_r2_score = gaussian_model.adjusted_r_squared(peaks_above_sn_in_search_window, fitting_results_reduced)
            if adjusted_r2_score >= best_score:
                best_score = adjusted_r2_score
                best_fitting_results = fitting_results_reduced
        
        print('\nBest results were found with r_square =', best_score)    
        """   
        
        plots_tables_obj.create_table_identified_masses(best_fitting_results, sample_name, bin_size_identified_masses)
        
        if len(best_fitting_results) == 0:
            print('\nNo mass shift detected for following condition: ' + sample_name)
            continue
  
        print('\n'+str(len(best_fitting_results))+' mass shifts are found.')  
            
        """
        mass_shifts = MassShifts(best_fitting_results)
        

        unmodified_species_mass = gaussian_model.determine_unmodified_species_mass(best_fitting_results, unmodified_species_mass_init, unmodified_species_mass_tol)
        if unmodified_species_mass == 0:
            print('\nUnmodified species could not be determined for following condition: ' + sample_name)
            continue

        mass_shifts.set_unmodified_species_mass(unmodified_species_mass)
        mass_shifts.calculate_mass_shifts()
        mass_shifts.refit_amplitudes(all_peaks_in_search_window, gaussian_model.stddev, sn_threshold)
        """
        
        best_fitting_results_refitted = gaussian_model.refit_amplitudes(all_peaks_in_search_window, best_fitting_results, sn_threshold)

        if calculate_mass_shifts == True:
            mass_shifts = MassShifts(best_fitting_results_refitted)
            unmodified_species_mass = gaussian_model.determine_unmodified_species_mass(best_fitting_results_refitted, unmodified_species_mass_init, unmodified_species_mass_tol)
        
            if unmodified_species_mass == 0:
                plots_tables_obj.plot_mass_shifts(data.raw_spectrum, all_peaks_in_search_window, mass_shifts, data.search_window_start_mass, data.search_window_end_mass, gaussian_model.stddev, sample_name)
                print('\nUnmodified species could not be determined for following condition: ' + sample_name)
                continue

            mass_shifts.set_unmodified_species_mass(unmodified_species_mass)
            mass_shifts.calculate_mass_shifts()
            
            plots_tables_obj.plot_mass_shifts(data.raw_spectrum, all_peaks_in_search_window, mass_shifts, data.search_window_start_mass, data.search_window_end_mass, gaussian_model.stddev, sample_name)
          
        else:
            mean = np.array(list(fitting_results.keys()))
            amplitude = np.array(list(fitting_results.values()))[:,0]
            plots_tables_obj.plot_masses(data.raw_spectrum, all_peaks_in_search_window, data.search_window_start_mass, data.search_window_end_mass, mean, amplitude, gaussian_model.stddev, sample_name)
       
        pp.savefig()
        
    pp.close()
    

    if calculate_mass_shifts == True:
        plots_tables_obj.add_mass_shifts()   
        plots_tables_obj.save_table_identified_masses(output_path)
    else:
        plots_tables_obj.save_table_identified_masses(output_path)


    identified_masses_table = plots_tables_obj.get_table_identified_masses()

    if (calculate_mass_shifts == True) and (determine_ptm_patterns == True):
        
        mass_mean = plots_tables_obj.identified_masses_table['mass mean'].values
        mass_shifts_mean = plots_tables_obj.identified_masses_table['mass shift'].values
        mass_std = plots_tables_obj.identified_masses_table['mass std'].values
        maximal_mass_error = [mass_mean[i]*mass_error*1.0e-6 if np.isnan(mass_std[i]) else mass_std[i] for i in range(len(mass_std))]
    
        print('\nSearching for PTM combinations:')
    
        all_ptm_patterns = mass_shifts.determine_all_ptm_patterns2(maximal_mass_error, mod, mass_shifts_mean)
        
        print('\n\nPTM patterns were found for', len(all_ptm_patterns), 'mass shifts.\n')  
        
        output_df, output_top_df = plots_tables_obj.save_identified_patterns(all_ptm_patterns, mod, output_path, top=top_results)

        plots_tables_obj.add_ptm_patterns(all_ptm_patterns, output_top_df, mod)
        
        plots_tables_obj.save_table_identified_masses(output_path)


    print(80*'-'+'\n\n')


    
    
"""



###
import matplotlib.pyplot as plt


peaks = data.get_peaks(min_peak_height=0.0)
peaks2 = data.remove_adjacent_peaks(peaks, 0.6)


plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color='gray')
#plt.plot(theoretical_distribution[:,0]+1, 4.5*theoretical_distribution[:,1], '--',  color='green')
plt.plot(peaks[:,0], peaks[:,1], '.',  color='red')
plt.plot(peaks2[:,0], peaks2[:,1], '.',  color='blue')
#plt.plot(np.array(list(fitting_results_reduced.keys())), np.array(list(fitting_results_reduced.values()))[:,0], '.b')
#plt.plot(np.array(list(fitting_results_reduced2.keys())), np.array(list(fitting_results_reduced2.values()))[:,0], '.g')
#plt.plot(mass_shifts.mean, mass_shifts.amplitude, '.g')

plt.ylabel('relative intensity (%)')
plt.show()
"""

