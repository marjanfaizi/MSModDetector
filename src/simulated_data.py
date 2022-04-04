#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 2021

@author: Marjan Faizi
"""

import glob
import pyopenms
import numpy as np
import pandas as pd
import re
import random

import utils 
import config
from error_estimation import ErrorEstimation


aa_sequence_str = utils.read_fasta('../data/fasta_files/P04637.fasta')
modifications_table =  pd.read_csv('../data/modifications/modifications_P04637.csv', sep=';')
modform_distribution =  pd.read_csv('../data/modifications/modform_distribution.csv', sep=',')

file_names = [file for file in glob.glob(config.file_names)] 

error_estimation = ErrorEstimation(file_names, config.conditions, config.replicates)
error_estimation.set_data_preprocessing_parameter(config.noise_level_fraction, config.distance_threshold_adjacent_peaks)
error_estimation.set_gaussian_model_parameter(config.stddev_isotope_distribution, config.unmodified_species_mass, config.pvalue_threshold)
error_estimation.estimate_error()

# TODO: find beta for exponential decay random number automatically
beta = 200
    
#####################################################################################################
############### Define methods to create mass spectrum from theoretical distribution ################
#####################################################################################################
def create_modified_seqeunce(modform, modifications_table, aa_sequence_str):
    mod_type =  re.findall(r'\[(.*?)\]', modform)
    mod_amount = re.findall(r'\d+', modform)

    modified_pos = list()
    mod_pos_dict = dict()
    for ix, m in enumerate(mod_type):
        sites = modifications_table[modifications_table["ptm_id"] == m]["site"].str.split(",").values[0]
        pos = [ match.start() for aa in sites for match in re.finditer(aa, aa_sequence_str) ]
        unoccupied_pos = list(set(pos).difference(set(modified_pos)))
        selected_pos = random.sample(unoccupied_pos, k=int(mod_amount[ix]))
        unimod_id = modifications_table[modifications_table["ptm_id"] == m]["unimod_id"].values[0]
        mod_pos_dict.update(dict(zip(selected_pos, [unimod_id]*int(mod_amount[ix]))))
        
    unimod_pos_dict = dict(sorted(mod_pos_dict.items()))
    modified_sequence_str = str()
    unimod_ids = list(unimod_pos_dict.values())
    for ix, pos in enumerate(unimod_pos_dict):
        if ix == 0:      
            modified_sequence_str += aa_sequence_str[:pos+1]            
        modified_sequence_str += "(UniMod:" + str(unimod_ids[ix]) + ")"         
        if ix == len(unimod_pos_dict)-1:
            modified_sequence_str += aa_sequence_str[pos+1:]
        else:
            pos_next = list(unimod_pos_dict)[ix+1]
            modified_sequence_str += aa_sequence_str[pos+1:pos_next+1]
    return modified_sequence_str

    
def seq_str_to_isotopic_dist(sequence_str, resolution, modus="coarse"):
    aa_sequence = pyopenms.AASequence.fromString(sequence_str)
    isotope_asarray = utils.get_theoretical_isotope_distribution(aa_sequence, resolution, modus=modus)
    return isotope_asarray


def determine_mass_spectrum_range(unmodified_species_distribution, max_mass_shift, spectrum_resolution):
    margin = 10
    mass_spectrum = np.arange(unmodified_species_distribution[0,0] - margin,
                              unmodified_species_distribution[-1,0] + max_mass_shift + margin, 
                              spectrum_resolution)
    return mass_spectrum
#####################################################################################################
#####################################################################################################


pattern_resolution = 100
max_mass_shift = 800
mass_grid_step = 0.02

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


# save mass spectrum as csv file
mass_spectrum = pd.DataFrame({"mass" : masses, "intensity" : intensities})
mass_spectrum.to_csv("../data/simulated_data_rep1.csv", index=False, header=False)


"""
from matplotlib import pyplot as plt

plt.figure(figsize=(7,3))
#plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color="0.3", linewidth=1)
plt.plot(masses, intensities*800, "k.-")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43630,44450))
plt.ylim(ymin=-0.005)
plt.tight_layout()
plt.show()
"""