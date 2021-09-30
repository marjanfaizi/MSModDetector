# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 2021

@author: Marjan Faizi
"""

import glob
import sys
import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#import seaborn as sns

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

output_fig = plt.figure(figsize=(14,7))
gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)


for rep in range(len(replicates)):
    if not file_names:
        print('\nFile does not exist.\n')
        sys.exit()
    
    file = file_names[rep]
    identified_masses_df = pd.read_csv(file, sep=',')

    for condition in range(len(masses_name)):
        color_of_sample = color_palette[condition]
        sample_name = masses_name[condition][7:]
        axes[condition].plot(identified_masses_df[masses_name[condition]], identified_masses_df[abundances_name[condition]],
                             label=sample_name+' ('+replicates[rep]+')', color=color_of_sample, marker=marker[rep], linestyle='None')
   

identified_masses_rep1_df = pd.read_csv(file_names[0], sep=',')
means_rep1 = identified_masses_rep1_df['average mass'].values

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



