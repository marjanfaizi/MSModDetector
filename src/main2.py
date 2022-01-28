#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2021

@author: Marjan Faizi
"""

import matplotlib.pyplot as plt
import numpy as np

from mass_spec_data import MassSpecData
from mass_shifts import MassShifts
from modifications import Modifications
import config

sample_name = '../data/raw_data/P04637\\uv_7hr_rep5.mzml'


# Calculate distance matrix for each mass peak
def create_distance_matrix(array):
    m, n = np.meshgrid(array, array)
    distance_matrix = abs(m-n)
    return distance_matrix


if __name__ == "__main__":
    
    mod = Modifications(config.modfication_file_name)
    mass_shifts = MassShifts()
    data = MassSpecData(sample_name)

    # Data preprocessing: select only peaks (centroid data) and normalize by maximum intensity within search window       
    all_peaks = data.picking_peaks()
    trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
    data.set_search_window_mass_range(config.mass_start_range, config.mass_end_range) 
    trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)

    max_intensity = trimmed_peaks_in_search_window[:,1].max()
    intensities_normalized = trimmed_peaks_in_search_window[:,1] / max_intensity
    trimmed_peaks_in_search_window[:,1] = intensities_normalized
    
    distances = create_distance_matrix(trimmed_peaks_in_search_window[:,0])
    upper_triangle = np.triu(distances)


plt.hist(upper_triangle[upper_triangle>0].flatten(), bins=900)



