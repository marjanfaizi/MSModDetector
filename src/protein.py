#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 5 2021

@author: Marjan Faizi
"""

import pyopenms
import numpy as np

class Protein(object):

    def __init__(self, fasta_file_name):
        self.isotope_pattern_resolution = 100
        self.sequence = self.__read_fasta_file(fasta_file_name)


    def __read_fasta_file(self, fasta_file_name):
        fasta_file_object = pyopenms.FASTAFile()
        fasta_file_entries = []
        fasta_file_object.load(fasta_file_name, fasta_file_entries)
        sequence = pyopenms.AASequence.fromString(fasta_file_entries[0].sequence)
        return sequence    
    
    def get_sequence(self):
        return self.sequence.toString()
    
    
    def get_theoretical_isotope_distribution(self, fasta_file_name):
        molecular_formula = self.sequence.getFormula()
        isotope_pattern_generator = pyopenms.CoarseIsotopePatternGenerator(self.isotope_pattern_resolution) 
        isotopes = molecular_formula.getIsotopeDistribution(isotope_pattern_generator)
        isotopes_aslist = []
        for iso in isotopes.getContainer():
            mass = iso.getMZ()
            intensity = iso.getIntensity()
            isotopes_aslist.append([mass, intensity])
        
        isotopes_asarray = np.asarray(isotopes_aslist)
        return isotopes_asarray

    def get_average_mass(self):
        return self.sequence.getAverageWeight()
    
    
    def get_monoisotopic_mass(self):
        return self.sequence.getMonoWeight()
    
    