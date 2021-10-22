# -*- coding: utf-8 -*-
"""
Created on Oct 20 2021

@author: Marjan Faizi
"""

import glob
import matplotlib.pyplot as plt
import seaborn as sns

from mass_spec_data import MassSpecData
import myconfig as config


plt.style.use("seaborn-pastel")

condition = "nutlin_only"
file_names = [file for file in glob.glob("../data/"+condition+"*.mzml")] 
replicates = ["rep1", "rep5", "rep6"]
max_mass_shift = 800


for ix, name in enumerate(file_names):
    data = MassSpecData(name)
    data.set_search_window_mass_range(config.start_mass_range, max_mass_shift)        

    max_intensity = data.intensities.max()
    data.convert_to_relative_intensities(max_intensity)
    rescaling_factor = max_intensity/100.0

    all_peaks = data.picking_peaks()
    trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
    trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
    
    sn_threshold = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()

    if replicates[ix] == "rep5":
        plt.plot(data.masses, data.intensities, "-", color="0.5")
        plt.plot(all_peaks[:,0], all_peaks[:,1], "r.")
        plt.plot(trimmed_peaks_in_search_window[:,0], trimmed_peaks_in_search_window[:,1], "g.")
        plt.xlim(data.search_window_start_mass-10, data.search_window_end_mass+10)
        plt.axhline(y=sn_threshold, c='0.1', lw=0.3)
        plt.axhline(y=2*sn_threshold, c='0.3', lw=0.3)
        plt.show()