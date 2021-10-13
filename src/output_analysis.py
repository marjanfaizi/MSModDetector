# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 2021

@author: Marjan Faizi
"""

import pandas as pd
import matplotlib.pyplot as plt
import myconfig as config
import numpy as np

output_fiel_name = "../output/mass_shifts.csv"

mass_shifts_df = pd.read_csv(output_fiel_name, sep=",")


output_fig = plt.figure(figsize=(14,7))
gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
axes = gs.subplots(sharex=True, sharey=True)



for cond in config.conditions:

    color_of_sample = config.color_palette[cond][0]
    order_in_plot = config.color_palette[cond][1]
    x = mass_shifts_df.filter(regex="masses "+cond).mean(axis=1)
    is_nan_mask = np.isnan(x)
    y = mass_shifts_df.filter(regex="rel. abundances "+cond).mean(axis=1)
    xerr = mass_shifts_df.filter(regex="masses "+cond).std(axis=1)
    yerr = mass_shifts_df.filter(regex="rel. abundances "+cond).std(axis=1)
    axes[order_in_plot].errorbar(x[~is_nan_mask], y[~is_nan_mask], yerr=yerr[~is_nan_mask], xerr=xerr[~is_nan_mask], 
                                  color= color_of_sample, marker=".", linestyle="none", label=cond)
    axes[order_in_plot].legend(fontsize=11, loc='upper right')
    means = mass_shifts_df["average mass"].values

    for m in means:
        axes[order_in_plot].axvline(x=m, c='0.3', ls='--', lw=0.3, zorder=0)

plt.xlabel('mass (Da)'); plt.ylabel('relative abundance')
plt.savefig('../output/comparison_all_replicates2.pdf', dpi=800)
plt.show()


