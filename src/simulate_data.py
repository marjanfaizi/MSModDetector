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
import sys
import random
import utils 



class SimulateData(object):
    """
    This class creates a mass spectrum from a theoretical modform distribution.
    Different error types and basal noise can be added to the theoretical mass spectrum.
    """

    
    def __init__(self, unmodified_sequence_str, modifications_table):
        self.unmodified_sequence_str = unmodified_sequence_str
        self.isotopic_pattern_resolution = 100
        self.isotopic_distribution_unmodified_species = self.seq_str_to_isotopic_dist(self.unmodified_sequence_str)
        self.modifications_table = modifications_table
        self.mass_grid_step = 0.02
        self.margin_mass_range = 20
        self.vertical_error_mean = 1.0
        self.vertical_error_std = None
        self.horizontal_error_beta = None
        self.basal_noise_beta = None
        self.peak_width_mean = None
        self.peak_width_std = None
        
        
    def set_peak_width_mean(self, peak_width_mean):
        self.peak_width_mean = peak_width_mean


    def reset_noise_error(self):
        self.vertical_error_std = None
        self.horizontal_error_beta = None
        self.basal_noise_beta = None
        self.peak_width_std = None


    def create_mass_spectrum(self, modform_distribution):       
        # create theoretical isotopic pattern for each modform in the theoretical modform distribution
        mass_grid = self.determine_mass_spectrum_range(modform_distribution)
        spectrum = np.array([]).reshape(0,2)
        for index, row in modform_distribution.iterrows():
            if row["PTM pattern"] == "0":
                isotopic_distribution_mod = self.seq_str_to_isotopic_dist(self.unmodified_sequence_str)
            else:
                modified_sequence_str = self.create_modified_seqeunce(row["PTM pattern"])
                isotopic_distribution_mod = self.seq_str_to_isotopic_dist(modified_sequence_str)
            
            scaling_factor = row["intensity"]/isotopic_distribution_mod[:,1].max()
            isotopic_distribution_mod[:,1] *= scaling_factor   
            spectrum = np.vstack((spectrum, isotopic_distribution_mod))

        masses_sorted_ix = np.argsort(spectrum, axis=0)[:,0]
        spectrum = spectrum[masses_sorted_ix]

        if self.basal_noise_beta != None and self.basal_noise_beta != 0:
            basal_noise = np.random.exponential(self.basal_noise_beta, size=spectrum.shape[0])
            spectrum[:,1] += basal_noise*scaling_factor

        if self.vertical_error_std != None and self.vertical_error_std != 0:
            vertical_error = np.random.normal(self.vertical_error_mean, self.vertical_error_std, size=spectrum.shape[0])
            spectrum[:,1] *= vertical_error

        if self.horizontal_error_beta != None and self.horizontal_error_beta != 0:       
            for index, val in enumerate(spectrum):
                horizontal_error = np.random.exponential(self.horizontal_error_beta)
                spectrum[index, 0] = val[0]+horizontal_error


        intensities = np.zeros(mass_grid.shape)
        
        for peak in spectrum:
            if self.peak_width_mean == None:
                sys.exit("Set value for peak_width_mean.")
            if self.peak_width_std != None and self.peak_width_std != 0:
                peak_width = random.gauss(self.peak_width_mean, self.peak_width_std)
            else:
                peak_width = self.peak_width_mean
            # Add gaussian peak shape centered around each theoretical peak
            intensities += peak[1] * np.exp(-0.5*((mass_grid - peak[0]) / peak_width)**2)

        return mass_grid, intensities


    def add_noise(self, vertical_error_std=None, horizontal_error_beta=None, basal_noise_beta=None, peak_width_std=None):
        # set the error/noise std/beta to the respective given value 
        # if no value is given, reset (important!) the value to None
        if vertical_error_std != None:
            self.vertical_error_std = vertical_error_std
        else:
            self.vertical_error_std = None

        if horizontal_error_beta != None:
            self.horizontal_error_beta = horizontal_error_beta
        else:
            self.horizontal_noise_std = None

        if basal_noise_beta != None:
            self.basal_noise_beta = basal_noise_beta
        else:
            self.basal_noise_beta = None

        if peak_width_std != None:
            self.peak_width_std = peak_width_std
        else:
            self.sigma_noise_std = None
            

    def save_mass_spectrum(self, masses, intensities, file_name):
        mass_spectrum_df = pd.DataFrame({"mass" : masses, "intensity" : intensities})
        mass_spectrum_df.to_csv(file_name, header=False, index=False)
        
        
    def determine_max_mass_shift(self, modform_distribution):
        max_mass = modform_distribution.mass.max()
        max_mass_shift = max_mass + self.margin_mass_range
        return max_mass_shift


    def create_modified_seqeunce(self, modform):
        mod_type =  re.findall(r'\[(.*?)\]', modform)
        mod_amount = re.findall(r'\d+', modform)
    
        modified_pos = list()
        mod_pos_dict = dict()
        for ix, m in enumerate(mod_type):
            sites = self.modifications_table[self.modifications_table["ptm_id"] == m]["site"].str.split(",").values[0]
            pos = [ match.start() for aa in sites for match in re.finditer(aa, self.unmodified_sequence_str) ]
            unoccupied_pos = list(set(pos).difference(set(modified_pos)))
            selected_pos = random.sample(unoccupied_pos, k=int(mod_amount[ix]))
            unimod_id = self.modifications_table[self.modifications_table["ptm_id"] == m]["unimod_id"].values[0]
            mod_pos_dict.update(dict(zip(selected_pos, [unimod_id]*int(mod_amount[ix]))))
            
        unimod_pos_dict = dict(sorted(mod_pos_dict.items()))
        modified_sequence_str = str()
        unimod_ids = list(unimod_pos_dict.values())
        for ix, pos in enumerate(unimod_pos_dict):
            if ix == 0:      
                modified_sequence_str += self.unmodified_sequence_str[:pos+1]            
            modified_sequence_str += "(UniMod:" + str(unimod_ids[ix]) + ")"         
            if ix == len(unimod_pos_dict)-1:
                modified_sequence_str += self.unmodified_sequence_str[pos+1:]
            else:
                pos_next = list(unimod_pos_dict)[ix+1]
                modified_sequence_str += self.unmodified_sequence_str[pos+1:pos_next+1]
        
        return modified_sequence_str


    def seq_str_to_isotopic_dist(self, aa_sequence_str, modus="coarse"):
        aa_sequence = pyopenms.AASequence.fromString(aa_sequence_str)
        isotope_asarray = utils.get_theoretical_isotope_distribution(aa_sequence, self.isotopic_pattern_resolution, modus=modus)
        return isotope_asarray


    def determine_mass_spectrum_range(self, modform_distribution):
        max_mass_shift = self.determine_max_mass_shift(modform_distribution)
        mass_spectrum = np.arange(self.isotopic_distribution_unmodified_species[0,0] - self.margin_mass_range,
                                  self.isotopic_distribution_unmodified_species[-1,0] + max_mass_shift + self.margin_mass_range, 
                                  self.mass_grid_step)
        return mass_spectrum


