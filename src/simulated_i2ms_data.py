#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 2021

@author: Marjan Faizi
"""

from brainpy import isotopic_variants
import pyopenms
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
import random
import utils 


#####################################################################################################
############################################### INPUT ###############################################
#####################################################################################################

aa_sequence_str = utils.read_fasta('../data/fasta_files/P04637.fasta')
modifications_table =  pd.read_csv('../data/modifications/modifications_P04637.csv', sep=';')

#####################################################################################################
#####################################################################################################



#####################################################################################################
################################# Generate theoretical distribution #################################
#####################################################################################################

modform_distribution =  pd.read_csv('../data/modifications/modform_distribution.csv', sep=';')
modform_distribution["relative intensity"] = modform_distribution["intensity"]/modform_distribution["intensity"].sum()
        
#####################################################################################################
#####################################################################################################


#####################################################################################################
######################## Create mass spectrum from theoretical distribution #########################
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
            pos_next = list(unimod_pos_dict)[ix+1]
            modified_sequence_str += aa_sequence_str[:pos+1] + "(UniMod:" + str(unimod_ids[ix]) + ")" + aa_sequence_str[pos+1:pos_next+1]
        elif ix == len(unimod_pos_dict)-1:
            modified_sequence_str += "(UniMod:" + str(unimod_ids[ix]) + ")"  + aa_sequence_str[pos+1:]   
        else:
            pos_next = list(unimod_pos_dict)[ix+1]
            modified_sequence_str += "(UniMod:" + str(unimod_ids[ix]) + ")" + aa_sequence_str[pos+1:pos_next+1]
            
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


isotopic_distribution_p53 = seq_str_to_isotopic_dist(aa_sequence_str, 100)
mass_spectrum = determine_mass_spectrum_range(isotopic_distribution_p53, 800.0, 0.02)


#####################

modform = modform_distribution.loc[12, "modform"]

for index, row in modform_distribution.iterrows():
    print(row)


modified_sequence_str = create_modified_seqeunce(modform, modifications_table, aa_sequence_str)
isotopic_distribution_mod = seq_str_to_isotopic_dist(modified_sequence_str, 100)


mz_grid = np.arange(isotopic_distribution_mod[0,0] - 1,
                    isotopic_distribution_mod[-1,0] + 1, 0.02)
intensity = np.zeros(mz_grid.shape)

noise_mean = isotopic_distribution_mod[:,1].mean()
noise_std = isotopic_distribution_mod[:,1].std()/2
noise = np.random.normal(noise_mean, noise_mean, size=isotopic_distribution_mod.shape[0])
isotopic_distribution_mod[:,1]+=noise
isotopic_distribution_mod[:,1] = np.abs(isotopic_distribution_mod[:,1])


sigma = 0.02
for peak in isotopic_distribution_mod:
    # Add gaussian peak shape centered around each theoretical peak
    intensity += peak[1] * np.exp(-(mz_grid - peak[0]) ** 2 / (2 * sigma)
            ) / (np.sqrt(2 * np.pi) * sigma)

# Normalize profile to 0-100
intensity = (intensity / intensity.max()) * 100




#plt.plot(isotopic_distribution_p53[:,0], isotopic_distribution_p53[:,1]*50, "r.-")
#plt.plot(isotopic_distribution_mod[:,0], isotopic_distribution_mod[:,1]*50, "b.-")
plt.plot(mz_grid, intensity, "g.-")
plt.xlabel("mass (Da)")
plt.ylabel("relative intensity (%)")
plt.show()

# create theoretical isotopic pattern for each modform in the theoretical modform distribution

# create basal noise level

# combine all isotopic patterns and noise level into one spectrum

# add horizontal and vertical noise 
from mass_spec_data import MassSpecData
sample_name = "xray_7hr_rep5"
file_name = "../data/raw_data/P04637/" + sample_name + ".mzml"
data = MassSpecData(file_name)

plt.figure(figsize=(7,3))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color="0.3", linewidth=1)
plt.plot(mz_grid+1.5, intensity*2, "g.-")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((3500,55000))
plt.ylim(ymin=-200)
plt.tight_layout()

plt.show()

#####################################################################################################
#####################################################################################################

import numpy as np
import matplotlib.pyplot as plt


nsample = 1024
## simulate a simple sinusoidal function
x1 = np.linspace(0, 100, nsample)
y=np.sin(x1)

sigma = 0.3
y =sigma * np.random.normal(size=nsample)
fig, ax = plt.subplots()
ax.plot(x1, y, label="Data")

sigma = 0.3
y =np.sin(x1)+sigma * np.random.normal(size=nsample)
fig, ax = plt.subplots()
ax.plot(x1, y, label="Data")

############################################# SONSTIGES #############################################
aa_sequence = pyopenms.AASequence.fromString(aa_sequence_str)
aa_sequence.getFormula()
molecular_formula = aa_sequence.getFormula()
molecular_formula.toString()
#C1898 H2980 N548 O592 S22

# Generate theoretical isotopic pattern
peptide = {'H': 2980, 'C': 1898, 'O': 592, 'N': 548, 'S':22}
theoretical_isotopic_cluster = isotopic_variants(peptide, npeaks=50, charge=1)
for peak in theoretical_isotopic_cluster:
    print(peak.mz, peak.intensity)



# produce a theoretical profile using a gaussian peak shape

mz_grid = np.arange(theoretical_isotopic_cluster[0].mz - 1,
                    theoretical_isotopic_cluster[-1].mz + 1, 0.02)
intensity = np.zeros_like(mz_grid)
sigma = 0.02
for peak in theoretical_isotopic_cluster:
    # Add gaussian peak shape centered around each theoretical peak
    intensity += peak.intensity * np.exp(-(mz_grid - peak.mz) ** 2 / (2 * sigma)
            ) / (np.sqrt(2 * np.pi) * sigma)

# Normalize profile to 0-100
intensity = (intensity / intensity.max()) * 100

# Plot the isotopic distribution
plt.plot(mz_grid, intensity, '.-')
plt.xlabel("mass (Da)")
plt.ylabel("relative intensity (%)")


#####################################################################################################

