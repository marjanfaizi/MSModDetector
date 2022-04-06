#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 5 2022

@author: Marjan Faizi
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import utils


aa_sequence_str = utils.read_fasta('../data/fasta_files/P04637.fasta')
modifications_table =  pd.read_csv('../data/modifications/modifications_P04637.csv', sep=';')
modform_distribution =  pd.read_csv('../data/modifications/modform_distribution.csv', sep=',')
error_estimate_table = pd.read_csv("../output/noise_distribution_table.csv")


output_filename = "../data/simulated_data_rep1.csv"






plt.figure(figsize=(7,3))
#plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:,1], '.-', color="0.3", linewidth=1)
#plt.plot(masses, intensities*800, "k.-")
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((43630,44450))
plt.ylim(ymin=-0.005)
plt.tight_layout()
plt.show()
