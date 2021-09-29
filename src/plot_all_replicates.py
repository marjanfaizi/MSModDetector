# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 2021

@author: Marjan Faizi
"""

import glob
import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#import seaborn as sns

import myconfig as config


file_names = [file for file in glob.glob('..\\output\\identified_masses_table_*_multiion.csv')] 

replicates = ['rep1', 'rep5', 'rep6']

output_fig = plt.figure(figsize=(14,7))
gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)


for ix in range(len(replicates)):
    file = file_names[ix]
    identified_masses = pd.read_csv(file, sep=',', header=None)


    color_of_sample = [value[0] for key, value in config.color_palette.items() if key in '\_'+sample_name][0]
    order_in_plot = [value[1] for key, value in config.color_palette.items() if key in '\_'+sample_name][0]
    #axes[order_in_plot].plot(data.masses, data.intensities*rescaling_factor, label=sample_name, color=color_of_sample)


for ax in axes:
    ax.set_xlim((config.start_mass_range, config.start_mass_range+config.max_mass_shift))
    ax.set_ylim((-0.01, 0.5))
    ax.yaxis.grid()
    ax.legend(fontsize=11, loc='upper right')
    ax.label_outer()
plt.xlabel('mass (Da)'); plt.ylabel('relative abundance')
output_fig.tight_layout()
plt.savefig('../output/comparison_of_replcitaes.pdf', dpi=800)
plt.show()