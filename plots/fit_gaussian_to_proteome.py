# -*- coding: utf-8 -*-
"""
Created on May 23 2022

@author: Marjan Faizi
"""

import utils
from scipy.stats import chisquare
from pyopenms import AASequence
import pandas as pd
import numpy as np


proteome = utils.read_fasta("../data/fasta_files/human_proteome_reviewed_23_05_22.fasta")

results = []

for protein_id in proteome:
    sequence = proteome[protein_id]
    monoisotopic_mass = AASequence.fromString(sequence).getMonoWeight()
    mean, std, amplitude = utils.isotope_distribution_fit_par(sequence, 100)
    if mean == 0 and  std == 0 and  amplitude == 0:
        continue
    else:
        distribution = utils.get_theoretical_isotope_distribution(AASequence.fromString(sequence), 100)
        non_zero_intensity_ix = np.where(distribution[:, 1] > 1e-10)[0]
        sample_size = distribution[non_zero_intensity_ix].shape[0]
        if sample_size >= 5:
            observed_masses = distribution[non_zero_intensity_ix, 0]
            observed_intensities = distribution[non_zero_intensity_ix, 1]
            predicted_intensities = utils.gaussian(observed_masses, amplitude, mean, std)
            score = chisquare(observed_intensities, f_exp=predicted_intensities, ddof=3)
            pvalue = score.pvalue
            chi_square = score.statistic
            
            results.append((protein_id, monoisotopic_mass, amplitude, mean, std, pvalue, chi_square))


results_df = pd.DataFrame(results, columns=("protein_id", "monoisotopic_mass", "amplitude", "mean", 
                                            "standard_deviation", "pvalue", "chi_square"))
results_df.to_csv("../output/fit_gaussian_to_proteome.csv", index=False)
