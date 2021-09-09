#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 5 2021

@author: Marjan Faizi
"""
#%matplotlib qt

import pyopenms
import numpy as np
import utils
import pandas as pd
import matplotlib.pyplot as plt



fasta_file_name = "/Users/marjanfaizi/Documents/Postdoc/Code/data/uniprot_human_proteome_reviewed_and_isoforms.fasta"
fasta_file_object = pyopenms.FASTAFile()
fasta_file_entries = []
fasta_file_object.load(fasta_file_name, fasta_file_entries)    

isotope_pattern_resolution = 100
mapping_protein_mass_to_stddev = {}

progess_bar = 0
print('\nCalculating the standard deviation for every protein in the human proteome database:')
for entry in fasta_file_entries:
    protein_id = entry.identifier
    sequence = pyopenms.AASequence.fromString(entry.sequence)
    if "X" in sequence.toString():  
        continue
        #print(protein_id)
    else:
        distribution = utils.get_theoretical_isotope_distribution(sequence, isotope_pattern_resolution)
        if np.isnan(distribution).any():
            continue
        else:
            average_mass = sequence.getAverageWeight()
            stddev = utils.estimate_stddev_from_isotope_distribution(distribution)
            mapping_protein_mass_to_stddev[protein_id] = [average_mass, stddev]
    progess_bar += 1
    utils.progress(progess_bar, len(fasta_file_entries))
print('\n')


protein_mass_stddev_table = pd.DataFrame.from_dict(mapping_protein_mass_to_stddev, orient='index')
protein_mass_stddev_table.reset_index(inplace=True)
protein_mass_stddev_table.columns = ["protein_id", "mass", "stddev"]
protein_mass_stddev_table.sort_values(by="mass", inplace=True, ignore_index=True)
protein_mass_stddev_table.to_csv("mass_to_stddev.csv", index=False)


protein_mass_stddev_table = pd.read_csv("mass_to_stddev.csv")

max_val_init = 5
steepness_init = 1e-5
midpoint_init = 1e4
initial_values = [max_val_init, steepness_init, midpoint_init]
popt_logistic_func = utils.fit_logistic_func(protein_mass_stddev_table.mass, protein_mass_stddev_table.stddev, initial_values)

plt.plot(protein_mass_stddev_table.mass, protein_mass_stddev_table.stddev, '.')
plt.plot(protein_mass_stddev_table.mass, utils.logistic_func(protein_mass_stddev_table.mass, *popt_logistic_func), '--')
plt.show()

