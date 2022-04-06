#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 2021

@author: Marjan Faizi
"""

import pyopenms
import numpy as np
import pandas as pd
import re
import random
import utils 



class SimulateData(object):
    """
    This class creates a mass spectrum from a theoretical modform distribution.
    Different noise types can be added to the theoretical mass spectrum.
    """

    
    def __init__(self, aa_sequence_str, modifications_table):
        self.aa_sequence_str = aa_sequence_str
        self.modifications_table = modifications_table
        self.isotopic_pattern_resolution = 100
        self.mass_grid_step = 0.02
        self.margin_mass_range = 10



    def create_mass_spectrum(self):
        pass


    def add_noise(self):
        pass


    def save_mass_spectrum(self, masses, intensities, file_name):
        mass_spectrum_df = pd.DataFrame({"mass" : masses, "intensity" : intensities})
        mass_spectrum_df.to_csv(file_name, index=False)
        


    def create_modified_seqeunce(self, modform):
        mod_type =  re.findall(r'\[(.*?)\]', modform)
        mod_amount = re.findall(r'\d+', modform)
    
        modified_pos = list()
        mod_pos_dict = dict()
        for ix, m in enumerate(mod_type):
            sites = self.modifications_table[self.modifications_table["ptm_id"] == m]["site"].str.split(",").values[0]
            pos = [ match.start() for aa in sites for match in re.finditer(aa, self.aa_sequence_str) ]
            unoccupied_pos = list(set(pos).difference(set(modified_pos)))
            selected_pos = random.sample(unoccupied_pos, k=int(mod_amount[ix]))
            unimod_id = self.modifications_table[self.modifications_table["ptm_id"] == m]["unimod_id"].values[0]
            mod_pos_dict.update(dict(zip(selected_pos, [unimod_id]*int(mod_amount[ix]))))
            
        unimod_pos_dict = dict(sorted(mod_pos_dict.items()))
        modified_sequence_str = str()
        unimod_ids = list(unimod_pos_dict.values())
        for ix, pos in enumerate(unimod_pos_dict):
            if ix == 0:      
                modified_sequence_str += self.aa_sequence_str[:pos+1]            
            modified_sequence_str += "(UniMod:" + str(unimod_ids[ix]) + ")"         
            if ix == len(unimod_pos_dict)-1:
                modified_sequence_str += self.aa_sequence_str[pos+1:]
            else:
                pos_next = list(unimod_pos_dict)[ix+1]
                modified_sequence_str += self.aa_sequence_str[pos+1:pos_next+1]
        
        return modified_sequence_str


    def seq_str_to_isotopic_dist(self, modus="coarse"):
        aa_sequence = pyopenms.AASequence.fromString(self.aa_sequence_str)
        isotope_asarray = utils.get_theoretical_isotope_distribution(aa_sequence, self.isotopic_pattern_resolution, modus=modus)
        return isotope_asarray


    def determine_mass_spectrum_range(self, unmodified_species_distribution, max_mass_shift):
        unmodified_species_distribution = self.eq_str_to_isotopic_dist()
        mass_spectrum = np.arange(unmodified_species_distribution[0,0] - self.margin_mass_range,
                                  unmodified_species_distribution[-1,0] + max_mass_shift + self.margin_mass_range, 
                                  self.mass_grid_step)
        return mass_spectrum




#####################################################################################################
#####################################################################################################




isotopic_distribution_p53 = seq_str_to_isotopic_dist(aa_sequence_str, pattern_resolution)
masses = determine_mass_spectrum_range(isotopic_distribution_p53, max_mass_shift, mass_grid_step)

# create theoretical isotopic pattern for each modform in the theoretical modform distribution
spectrum = np.array([]).reshape(0,2)
for index, row in modform_distribution.iterrows():
    modified_sequence_str = create_modified_seqeunce(row["modform"], modifications_table, aa_sequence_str)
    isotopic_distribution_mod = seq_str_to_isotopic_dist(modified_sequence_str, 100)
    scaling_factor = row["relative intensity"]/isotopic_distribution_mod[:,1].max()
    isotopic_distribution_mod[:,1] *= scaling_factor    
    vertical_error = np.random.normal(error_estimation.vertical_error.mean(), error_estimation.vertical_error.std()/8, size=isotopic_distribution_mod.shape[0])
    isotopic_distribution_mod[:,1] += vertical_error
    basal_noise = np.random.exponential(1/beta, size=isotopic_distribution_mod.shape[0])
    isotopic_distribution_mod[:,1] += basal_noise
    isotopic_distribution_mod[isotopic_distribution_mod[:,1]<0,1] = 0

    for index, val in enumerate(isotopic_distribution_mod):
        horizontal_error = random.gauss(error_estimation.horizontal_error.mean(), error_estimation.horizontal_error.std()/8)
        isotopic_distribution_mod[index, 0] = val[0]+horizontal_error
    
    spectrum = np.vstack((spectrum, isotopic_distribution_mod))

masses_sorted_ix = np.argsort(spectrum, axis=0)[:,0]
spectrum = spectrum[masses_sorted_ix]

intensities = np.zeros(masses.shape)

for peak in spectrum:
    sigma = 0.28
#    sigma = random.gauss(error_estimation.sigma_sinlge_peak.mean(), error_estimation.sigma_sinlge_peak.std()/8)
    # Add gaussian peak shape centered around each theoretical peak
    intensities += peak[1] * np.exp(-0.5*((masses - peak[0]) / sigma)**2)




