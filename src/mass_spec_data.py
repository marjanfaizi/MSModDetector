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
        
        self.mass_error = 3.0
        self.search_window_start_mass = self.raw_spectrum[0,0]
        self.search_window_end_mass = self.raw_spectrum[-1,0]
        self.max_mass_shift = 500.0


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


    def set_mass_error(self, mass_error):
        self.mass_error = mass_error


    def set_max_mass_shift(self, max_mass_shift):
        self.max_mass_shift = max_mass_shift


    def set_search_window_mass_range(self, start_mass):
        self.search_window_start_mass = start_mass
        self.search_window_end_mass = start_mass + self.max_mass_shift


    def determine_search_window(self, peaks):
        start_indices = self.__find_peaks_by_mass(peaks, self.search_window_start_mass)
        if len(start_indices) != 0:
            end_indices =  self.__find_peaks_by_mass(peaks, self.search_window_end_mass)
            search_window_indices = np.arange(start_indices[0], end_indices[-1]+1)
            peaks_in_search_window = peaks[search_window_indices]
        else:
            peaks_in_search_window = np.array([])
        return peaks_in_search_window


    def __find_peaks_by_mass(self, peaks, mass):
        peaks_mass = peaks[:,0]
        mass_tolerance_Da = mass*self.mass_error*1.0e-6
        found_masses_indices = []
        while ( len(found_masses_indices) == 0  and mass_tolerance_Da < self.max_mass_shift):
            found_masses_indices = np.argwhere((peaks_mass<=mass+mass_tolerance_Da) & (peaks_mass>=mass-mass_tolerance_Da))[:,0]
            mass_tolerance_Da = mass_tolerance_Da*1.1

        return found_masses_indices


    def convert_to_relative_intensities(self, max_intensity):
        relative_intensities = 100 * self.intensities / max_intensity
        self.intensities = relative_intensities


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
    
