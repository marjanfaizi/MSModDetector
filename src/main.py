#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2021

@author: Marjan Faizi
"""

#%matplotlib auto
 
import glob
import sys
import re
from matplotlib.backends.backend_pdf import PdfPages

from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel
from mass_shifts import MassShifts
from modifications import Modifications
import utils


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
#path = '/Users/marjanfaizi/Documents/Postdoc/Data/TopDown/06_22_2021/'
path = '/Users/marjanfaizi/Documents/Postdoc/Data/TopDown/MultiIonFiteringComp/'
# *_01_Profile.mzml, *_02_Profile.mzml, *_01_CentPlus1.mzml, *_02_CentPlus1.mzml
# *_01_Profile_MultiIonFiltered.mzml, *_02_Profile_MultiIonFiltered.mzml, *_01_CentPlus1_MultiIonFiltered.mzml, *_02_CentPlus1_MultiIonFiltered.mzml
 
file_names = [file for file in glob.glob(path+'*_01_Profile.mzml')] 
regex_extract_output_name = 'MCF7_(.*)_Profile'
modfication_file_name = '/Users/marjanfaizi/Documents/Postdoc/Code/data/modifications_P04637.csv'
max_mass_shift = 900.0 #Da
start_mass_range = 43750.0 #Da
unmodified_species_mass_init = 43770.0 #Da
unmodified_species_mass_tol = 5.0 #Da

mass_error = 5.0 #ppm
pvalue_threshold = 0.99
distance_threshold_adjacent_peaks = 0.6
bin_size_identified_masses = 5.0 #Da
calculate_mass_shifts = True
determine_ptm_patterns = True

##################################################################################################### 

if __name__ == '__main__':
    
    print('\n'+'-'*80) 
        
    if not file_names:
        print('\nFile does not exist.')
        sys.exit()
    
    mod = Modifications(modfication_file_name)
    mod.get_modification_masses()

    mass_shifts = MassShifts()
     
    pp = PdfPages('../output/identified_masses.pdf')
    
    progress_bar_mass_shifts = 0

    for file_name in file_names:
                
        sample_name = re.search(regex_extract_output_name, file_name).group(1).lower()
    
        data = MassSpecData(file_name)
        data.set_mass_error(mass_error)
        data.set_max_mass_shift(max_mass_shift)
        data.set_search_window_mass_range(start_mass_range)  
        
        max_intensity = data.intensities.max()
        data.convert_to_relative_intensities(max_intensity)
        
        all_peaks = data.picking_peaks()
        trimmed_peaks = data.remove_adjacent_peaks(all_peaks, distance_threshold_adjacent_peaks)   
        trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
                
        if not trimmed_peaks_in_search_window.size:
            print('\nNo peaks could be detected within the search window for the following condition: ' + sample_name)
            continue
    
        # TODO: is there another way to calculate th s/n ratio?
        sn_threshold = trimmed_peaks_in_search_window[:,1].mean()/trimmed_peaks_in_search_window[:,1].std()   
        
        peaks_above_sn = data.picking_peaks(min_peak_height=sn_threshold)
        trimmed_peaks_above_sn  = data.remove_adjacent_peaks(peaks_above_sn, distance_threshold_adjacent_peaks)  
        trimmed_peaks_above_sn_in_search_window = data.determine_search_window(trimmed_peaks_above_sn)
    
        if not trimmed_peaks_above_sn_in_search_window.size:
            print('\nNo peaks above the SN threshold could be detected within the search window for the following condition: ' + sample_name)
            continue
            
        # 1. ASSUMPTION: The isotopic distribution follows a normal distribution.
        # 2. ASSUMPTION: The standard deviation does not change when modifications are included to the protein mass. 
        gaussian_model = GaussianModel()
        gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks, trimmed_peaks_above_sn_in_search_window)
        gaussian_model.filter_fitting_results(pvalue_threshold)
        gaussian_model.refit_amplitudes(trimmed_peaks_in_search_window, sn_threshold)

        mass_shifts.create_table_identified_masses(gaussian_model.means, sample_name, bin_size_identified_masses)
        
        if not gaussian_model.fitting_results:
            print('\nNo masses detected for the following condition: ' + sample_name)
            continue
        
        if calculate_mass_shifts == True:
            unmodified_species_mass = utils.determine_unmodified_species_mass(gaussian_model.means, unmodified_species_mass_init, unmodified_species_mass_tol)
        
            if unmodified_species_mass == 0:
                utils.plot_spectra(data, trimmed_peaks_in_search_window, gaussian_model, sample_name)
                print('\nUnmodified species could not be determined for the following condition: ' + sample_name)
                continue

            utils.plot_spectra(data, trimmed_peaks_in_search_window, gaussian_model, sample_name, unmodified_species_mass)
            
        else:
            utils.plot_spectra(data, trimmed_peaks_in_search_window, gaussian_model, sample_name)
        
        progress_bar_mass_shifts += 1        
        utils.progress(progress_bar_mass_shifts, len(file_names))
        
        pp.savefig()
    
    pp.close()
    
    if calculate_mass_shifts == True:
        masses_means = mass_shifts.identified_masses_table['mass mean'].values
        unmodified_species_mass = utils.determine_unmodified_species_mass(masses_means, unmodified_species_mass_init, unmodified_species_mass_tol)        
        mass_shifts.add_mass_shifts(unmodified_species_mass)   
        mass_shifts.save_table_identified_masses('../output/')
    else:
        mass_shifts.save_table_identified_masses('../output/')

    if (calculate_mass_shifts == True) and (determine_ptm_patterns == True):
        maximal_mass_error = mass_shifts.estimate_maximal_mass_error(mass_error)
        
        print('\nSearching for PTM combinations:')
    
        mass_shifts.determine_ptm_patterns(mod, maximal_mass_error)
        
        print('\n\nPTM patterns were found for', len(mass_shifts.ptm_patterns_table.shape[0]), 'mass shifts.\n')  
        
        mass_shifts.add_ptm_patterns_to_table()
        mass_shifts.save_table_identified_masses('../output/')
    
    print(80*'-'+'\n\n')

    

#import multiprocessing as mp
#print("Number of processors: ", mp.cpu_count())


"""
import matplotlib.pyplot as plt

peaks = data.get_peaks(min_peak_height=0.0)
peaks2 = data.remove_adjacent_peaks(peaks, 0.6)

plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color='gray')
#plt.plot(theoretical_distribution[:,0], theoretical_distribution[:,1]*450, '--',  color='green')
#plt.plot(peaks[:,0], peaks[:,1], '.',  color='red')
plt.plot(peaks2[:,0], peaks2[:,1], '.',  color='blue')
#plt.plot(np.array(list(fitting_results_reduced.keys())), np.array(list(fitting_results_reduced.values()))[:,0], '.b')
#plt.plot(np.array(list(fitting_results_reduced2.keys())), np.array(list(fitting_results_reduced2.values()))[:,0], '.g')
#plt.plot(mass_shifts.mean, mass_shifts.amplitude, '.g')
plt.ylabel('relative intensity (%)')
plt.show()
"""
