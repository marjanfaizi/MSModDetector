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
    Pekas are picked from the raw mass spectrum. All subsequent analyses is performed using peaks.
    'Shoulder' peaks are removed to improve analyses.    
    """
    
    def __init__(self, data_file_name):
        self.data_file_seperator = ","
        self.raw_spectrum  = self.__read_data(data_file_name)
        self.masses = self.raw_spectrum[:,0]
        self.intensities = self.raw_spectrum[:,1]
        
        self.mass_error = 10.0 # ppm
        self.search_window_start_mass = self.raw_spectrum[0,0]
        self.search_window_end_mass = self.raw_spectrum[-1,0]


    def __read_data(self, data_file_name):
        if ".txt" in data_file_name or ".csv" in data_file_name:
            try:
                spectrum = pd.read_csv(data_file_name, sep=self.data_file_seperator, header=None)
                spectrum_asarray = np.asarray(spectrum)
                return spectrum_asarray
            except FileNotFoundError:
                print('File does not exist.')
                sys.exit()
        elif ".mzml" in data_file_name:
            try:
                openms_object = pyopenms.MSExperiment()
                pyopenms.MzMLFile().load(data_file_name, openms_object)
                spectrum = openms_object.getSpectrum(0).get_peaks()  
                spectrum_asarray = np.transpose(np.asarray(spectrum))
                return spectrum_asarray
            except FileNotFoundError:
                print('File does not exist.')
                sys.exit()


    def set_search_window_mass_range(self, start_mass, end_mass):
        self.search_window_start_mass = start_mass
        self.search_window_end_mass = end_mass


    def determine_search_window(self, peaks):
        start_index = self.find_nearest_mass_idx(peaks, self.search_window_start_mass)
        end_index =  self.find_nearest_mass_idx(peaks, self.search_window_end_mass)
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
        peaks_index, _ = find_peaks(self.intensities, height=min_peak_height)
        peaks_intensity = self.intensities[peaks_index]
        peaks_mass = self.masses[peaks_index]
        peaks = np.transpose(np.vstack((peaks_mass, peaks_intensity)))
        return peaks


    def remove_adjacent_peaks(self, peaks, distance_threshold):
        peaks_shifted = np.vstack((peaks[1:], peaks[0]))
        distances = np.abs(peaks_shifted[:,0] - peaks[:,0])
        adjacent_peaks_ix = np.where(distances < distance_threshold)[0]
        adjacent_intensities = np.vstack((peaks[adjacent_peaks_ix,1], 
                                          peaks[adjacent_peaks_ix+1,1]))
        remove_intensities_ix = np.argmin(adjacent_intensities, axis=0)
        removed_adjacent_peaks = np.delete(peaks, adjacent_peaks_ix+remove_intensities_ix, axis=0)
        if len(removed_adjacent_peaks) < len(peaks):
            return self.remove_adjacent_peaks(removed_adjacent_peaks, distance_threshold)
        else:
            return removed_adjacent_peaks    
    
