# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 2021

@author: Marjan Faizi
"""

import glob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import myconfig as config


#file_names = [file for file in glob.glob('..\\output\\identified_masses_table_*_multiion.csv')] 
file_names = [file for file in glob.glob('../output/identified_masses_table_*_multiion.csv')] 
masses_name = ['masses nutlin_7hr', 'masses xray_2hr', 'masses xray_7hr', 'masses xray-nutlin_7hr', 'masses uv_7hr']
abundances_name =['rel. abundances nutlin_7hr', 'rel. abundances xray_2hr', 'rel. abundances xray_7hr',
                  'rel. abundances xray-nutlin_7hr', 'rel. abundances uv_7hr']
color_palette = ['skyblue', 'yellowgreen', 'lightseagreen', 'chocolate', 'mediumpurple']
replicates = ['rep1', 'rep5', 'rep6']
marker = ['o', 'X', 'v']
linestyle = ['dashed', 'solid', 'dashdot']

def create_distance_matrix(array):
    m, n = np.meshgrid(array, array)
    distance_matrix = abs(m-n)
    arbitrary_high_number = 100
    np.fill_diagonal(distance_matrix, arbitrary_high_number)
    return distance_matrix
    

            
rep_df = pd.DataFrame() 
           
for rep in range(len(replicates)):
    if not file_names:
        print('\nFile does not exist.\n')
        sys.exit()
    file = file_names[rep]
    identified_masses_df = pd.read_csv(file, sep=',')
    identified_masses_df['mass_index'] = identified_masses_df['average mass'].astype(int)
    if rep_df.empty:
        rep_df = pd.DataFrame(columns=identified_masses_df.columns)

    rep_df = rep_df.merge(identified_masses_df, how='outer', left_on='mass_index', right_on='mass_index', suffixes=('', '_'+replicates[rep]))

    """
    for condition in range(len(masses_name)):
        color_of_sample = color_palette[condition]
        sample_name = masses_name[condition][7:]
        axes[condition].plot(identified_masses_df[masses_name[condition]], identified_masses_df[abundances_name[condition]],
                             label=sample_name+' ('+replicates[rep]+')', color=color_of_sample, marker=marker[rep], linestyle='None')
   """
   
   
rep_df.dropna(axis=1, how='all', inplace=True)
rep_df.drop(list(map('average mass_'.__add__,replicates)), axis=1, inplace=True)   
rep_df.drop(list(map('PTM pattern (P,Ac,Me,Ox,-H2O,Na)_'.__add__,replicates)), axis=1, inplace=True)   
rep_df.drop(list(map('mass shift_'.__add__,replicates)), axis=1, inplace=True)  
rep_df.drop(list(map('mass std_'.__add__,replicates)), axis=1, inplace=True)  

rep_df['average mass'] = rep_df.filter(regex='masses').mean(axis=1).values
rep_df.sort_values(by='average mass', inplace=True)
rep_df.reset_index(drop=True, inplace=True)
distance_matrix = create_distance_matrix(rep_df['average mass'].values)
while (distance_matrix < config.bin_size_mass_shifts).any():
    indices_to_combine = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
    rep_df.iloc[indices_to_combine[0]] = rep_df.iloc[indices_to_combine[0]:indices_to_combine[1]+1].max()
    rep_df.drop(indices_to_combine[1], inplace=True)
    rep_df['average mass'] = rep_df.filter(regex='masses').mean(axis=1).values
    rep_df.sort_values(by='average mass', inplace=True)
    rep_df.reset_index(drop=True, inplace=True)
    distance_matrix = create_distance_matrix(rep_df['average mass'].values)
    
    
rep_df['average mass'] = rep_df.filter(regex='masses').mean(axis=1).values    
rep_df['mass std'] = rep_df.filter(regex='masses').std(axis=1).values
rep_df.sort_values(by='average mass', inplace=True)
rep_df.reset_index(drop=True, inplace=True)


output_fig = plt.figure(figsize=(14,7))
gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)

   
   
for ax in axes:
    ax.set_xlim((config.start_mass_range, config.start_mass_range+config.max_mass_shift+100))
    ax.set_ylim((-0.01, 0.26))
    ax.yaxis.grid()
    ax.legend(fontsize=11, loc='upper right')
    for m in means_rep1:
        ax.axvline(x=m, c='0.3', ls='--', lw=0.3, zorder=0)
    ax.label_outer()
plt.xlabel('mass (Da)'); plt.ylabel('relative abundance')
output_fig.tight_layout()
plt.savefig('../output/comparison_of_replcitaes.pdf', dpi=800)
plt.show()



