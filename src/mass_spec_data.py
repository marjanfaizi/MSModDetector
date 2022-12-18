#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 2 2021

@author: Marjan Faizi
"""

import sys
import pyopenms
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


class MassSpecData(object):
    """
    This class reads mass spectra in the following formats: .txt, .csv, and .mzMl 
    Peaks are picked from the raw mass spectrum. All subsequent analyses is performed using peaks.
    'Shoulder' peaks are removed to improve analyses.    
    """
    
    def __init__(self, raw_spectrum=None):
        self.raw_spectrum = raw_spectrum
        self.data_file_seperator = ","
        self.mass_error = 10.0 # ppm
        self.distance_threshold = 0.6


    def add_raw_spectrum(self, data_file_name):
        if ".txt" in data_file_name or ".csv" in data_file_name:
            try:
                raw_spectrum = pd.read_csv(data_file_name, sep=self.data_file_seperator, header=None)
                self.raw_spectrum = np.asarray(raw_spectrum)
            except FileNotFoundError:
                print('File does not exist.')
                sys.exit()
        elif ".mzml" in data_file_name  or ".mzML" in data_file_name:
            try:
                openms_object = pyopenms.MSExperiment()
                pyopenms.MzMLFile().load(data_file_name, openms_object)
                raw_spectrum = openms_object.getSpectrum(0).get_peaks()  
                self.raw_spectrum = np.transpose(np.asarray(raw_spectrum))
            except FileNotFoundError:
                print('File does not exist.')
                sys.exit()


    def set_mass_range_of_interest(self, start_mass, end_mass):
        self.search_window_start = start_mass
        self.search_window_end = end_mass


    def preprocess_peaks(self, peaks):
        trimmed_peaks = self.remove_adjacent_peaks(peaks)   
        trimmed_peaks_in_search_window = self.determine_search_window(trimmed_peaks)
        trimmed_peaks_in_search_window[:,1] = self.normalize_intensities(trimmed_peaks_in_search_window[:,1])
        return trimmed_peaks_in_search_window


    def determine_search_window(self, peaks):
        start_index = self.find_nearest_mass_idx(peaks, self.search_window_start)
        end_index =  self.find_nearest_mass_idx(peaks, self.search_window_end)
        search_window_indices = np.arange(start_index, end_index+1)
        peaks_in_search_window = peaks[search_window_indices]
        return peaks_in_search_window


    def find_nearest_mass_idx(self, peaks, mass):
        peaks_mass = peaks[:,0]
        nearest_mass_idx = np.abs(peaks_mass - mass).argmin()
        return nearest_mass_idx


    def normalize_intensities(self, intensities):
        max_intensity = intensities.max()
        intensities_normalized = intensities / max_intensity
        self.rescaling_factor = max_intensity
        return intensities_normalized


    def picking_peaks(self, min_peak_height=None):
        masses = self.raw_spectrum[:,0]
        intensities = self.raw_spectrum[:,1]
        peaks_index, _ = find_peaks(intensities, height=min_peak_height)
        peaks = np.transpose(np.vstack((masses[peaks_index], intensities[peaks_index])))
        return peaks


    def remove_adjacent_peaks(self, peaks):
        peaks_shifted = np.vstack((peaks[1:], peaks[0]))
        distances = np.abs(peaks_shifted[:,0] - peaks[:,0])
        adjacent_peaks_ix = np.where(distances < self.distance_threshold)[0]
        adjacent_intensities = np.vstack((peaks[adjacent_peaks_ix,1], 
                                          peaks[adjacent_peaks_ix+1,1]))
        remove_intensities_ix = np.argmin(adjacent_intensities, axis=0)
        removed_adjacent_peaks = np.delete(peaks, adjacent_peaks_ix+remove_intensities_ix, axis=0)
        if len(removed_adjacent_peaks) < len(peaks):
            return self.remove_adjacent_peaks(removed_adjacent_peaks)
        else:
            return removed_adjacent_peaks    


