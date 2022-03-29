#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 2021

@author: Marjan Faizi
"""

#from brainpy import isotopic_variants
import pyopenms
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
import re
import random
import utils 
import config
from mass_spec_data import MassSpecData
from gaussian_model import GaussianModel


#####################################################################################################
############################################### INPUT ###############################################
#####################################################################################################
aa_sequence_str = utils.read_fasta('../data/fasta_files/P04637.fasta')
modifications_table =  pd.read_csv('../data/modifications/modifications_P04637.csv', sep=';')
modform_distribution =  pd.read_csv('../data/modifications/modform_distribution.csv', sep=',')
#####################################################################################################
#####################################################################################################


#####################################################################################################
################### Analyzing horizontal and vertical error in experimental data ####################
#####################################################################################################
sample_name = "xray_7hr_rep5"
file_name = "../data/raw_data/P04637/" + sample_name + ".mzml"
data = MassSpecData(file_name)     
data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range)   
data_in_search_window = data.determine_search_window(data.raw_spectrum)    
all_peaks = data.picking_peaks()
trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)

#intensity = np.zeros(data_in_search_window.shape[0])
sigmas = []
for peak in trimmed_peaks_in_search_window:
    window_size = 0.5
    selected_region_ix = np.argwhere((data_in_search_window[:,0]<=peak[0]+window_size) & (data_in_search_window[:,0]>=peak[1]-window_size))[:,0]
    selected_region = data_in_search_window[selected_region_ix]         
    sigma_init = 0.2
    optimized_param, _ = optimize.curve_fit(lambda x, sigma: utils.gaussian(x, peak[1], peak[0], sigma), 
                                            selected_region[:,0], selected_region[:,1], maxfev=100000,
                                            p0=[sigma_init], bounds=(0, 0.5))
    sigma = optimized_param[0]
    sigmas.append(sigma)
#    intensity += peak[1] *  np.exp(-0.5*((data_in_search_window[:,0] -  peak[0]) / sigma)**2)

sigmas = np.sort(sigmas)
sigmas = sigmas[sigmas<0.49]

"""
plt.plot(trimmed_peaks_in_search_window[:,0], sigmas, "r.") 
plt.show() 

plt.hist(sigmas, bins=100)
plt.plot(sigmas, utils.gaussian(sigmas, 20, sigmas.mean(), sigmas.std()), "r.-") 
plt.show() 

#plt.figure(figsize=(7,3))
#plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color="0.3", linewidth=1)
#plt.plot(data_in_search_window[:,0], intensity, "g.-")
#plt.xlabel("mass (Da)")
#plt.ylabel("intensity (a.u.)")
#plt.xlim((3500,55000))
#plt.ylim(ymin=-200)
#plt.tight_layout()
#plt.show()
"""

# horizontal error
dist = trimmed_peaks_in_search_window[1:,0]-trimmed_peaks_in_search_window[:-1,0]
dist = np.sort(dist[dist<2])

"""
plt.hist(dist, bins=20)
plt.plot(dist, utils.gaussian(dist, 80, dist.mean(), dist.std()), "r.-") 
"""

trimmed_peaks_in_search_window[:,1] = data.normalize_intensities(trimmed_peaks_in_search_window[:,1])
noise_level = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()
gaussian_model = GaussianModel("xray_7hr", config.stddev_isotope_distribution)
gaussian_model.determine_adaptive_window_sizes(config.unmodified_species_mass)
gaussian_model.fit_gaussian_to_single_peaks(trimmed_peaks_in_search_window, noise_level, config.pvalue_threshold)
gaussian_model.remove_overlapping_fitting_results()
gaussian_model.refit_amplitudes(trimmed_peaks_in_search_window, noise_level)

start_ix = np.abs(trimmed_peaks_in_search_window[:,0]-43745).argmin()
end_ix = np.abs(trimmed_peaks_in_search_window[:,0]-43805).argmin()
trimmed_peaks_in_search_window[start_ix:end_ix+1,0]

x_gauss_func = trimmed_peaks_in_search_window[start_ix:end_ix+1,0]
y_gauss_func = utils.multi_gaussian(x_gauss_func, gaussian_model.fitting_results["amplitude"], gaussian_model.fitting_results["mean"], gaussian_model.stddev)

vertical_error = y_gauss_func - trimmed_peaks_in_search_window[start_ix:end_ix+1,1]
vertical_error = np.sort(vertical_error)

"""
plt.hist(vertical_error, bins=50)
plt.plot(vertical_error, utils.gaussian(vertical_error, 5, vertical_error.mean(), vertical_error.std()), "r.-") 
"""

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


isotopic_distribution_p53 = seq_str_to_isotopic_dist(aa_sequence_str, 100)
masses = determine_mass_spectrum_range(isotopic_distribution_p53, 800.0, 0.02)

# create theoretical isotopic pattern for each modform in the theoretical modform distribution
spectrum = np.array([]).reshape(0,2)
for index, row in modform_distribution.iterrows():
    modified_sequence_str = create_modified_seqeunce(row["modform"], modifications_table, aa_sequence_str)
    isotopic_distribution_mod = seq_str_to_isotopic_dist(modified_sequence_str, 100)
    scaling_factor = row["relative intensity"]/isotopic_distribution_mod[:,1].max()
    isotopic_distribution_mod[:,1] *= scaling_factor    
#    noise = np.random.normal(vertical_error.mean(), vertical_error.std()/6, size=isotopic_distribution_mod.shape[0])
    #isotopic_distribution_mod[:,1] += noise
    #isotopic_distribution_mod[:,1] = np.abs(isotopic_distribution_mod[:,1])
    isotopic_distribution_mod[isotopic_distribution_mod[:,1]<0,1] = 0

#    for index, val in enumerate(isotopic_distribution_mod):
#        horizontal_error = random.gauss(dist.mean(), dist.std()/6)-dist.mean()
#        isotopic_distribution_mod[index, 0] = val[0]+horizontal_error
    
    spectrum = np.vstack((spectrum, isotopic_distribution_mod))

masses_sorted_ix = np.argsort(spectrum, axis=0)[:,0]
spectrum = spectrum[masses_sorted_ix]

intensities = np.zeros(masses.shape)

for peak in spectrum:
#    sigma = random.gauss(sigmas.mean(), sigmas.std()/4)
    sigma = sigmas.mean()
    # Add gaussian peak shape centered around each theoretical peak
    intensities += peak[1] * np.exp(-0.5*((masses - peak[0]) / sigma)**2)


plt.figure(figsize=(7,3))
#plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color="0.3", linewidth=1)
plt.plot(masses, intensities, "k-")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43630,44450))
plt.ylim(ymin=-0.005)
plt.tight_layout()
plt.show()

#####################################################################################################
#####################################################################################################


#####################################################################################################
################################## Save mass spectrum as csv file ###################################
#####################################################################################################

mass_spectrum = pd.DataFrame({"mass" : masses, "intensity" : intensities})
mass_spectrum.to_csv("../data/simulated_data_rep1.csv", index=False, header=False)
#####################################################################################################
#####################################################################################################


"""
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



masses = []
for ix, row in modform_distribution.iterrows():
    modform = row["modform"]
    mod_type =  re.findall(r'\[(.*?)\]', modform)
    mod_amount = re.findall(r'\d+', modform)
    print(mod_type)
    mass = 0
    for ix, m in enumerate(mod_type):
        print(modifications_table[modifications_table["ptm_id"] == m].ptm_mass.values[0])
        print(type(mod_amount[ix]))
        mass += modifications_table[modifications_table["ptm_id"] == m].ptm_mass.values[0] * int(mod_amount[ix])
    masses.append(mass)

modform_distribution["mass"] = masses
modform_distribution.to_csv('../data/modifications/modform_distribution.csv', index=False)
#####################################################################################################
"""
