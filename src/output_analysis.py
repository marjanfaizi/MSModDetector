# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 2021

@author: Marjan Faizi
"""

import pandas as pd
import myconfig as config
import numpy as np


output_fiel_name = "../output/mass_shifts.csv"

mass_shifts_df = pd.read_csv(output_fiel_name, sep=",")

keep_mass_shifts_indices = np.arange(len(mass_shifts_df))
for cond in config.conditions:
    missing_masses_count = mass_shifts_df.filter(regex="masses .*"+cond).isnull().sum(axis=1)
    keep_mass_shifts_indices = np.intersect1d(keep_mass_shifts_indices, mass_shifts_df[missing_masses_count>=2].index)
                                              
                                              

mass_shifts_df_reduced = mass_shifts_df.drop(index=keep_mass_shifts_indices)
mass_shifts_df_reduced.to_csv('../output/mass_shifts_reduced_table.csv', sep=',', index=False)  

