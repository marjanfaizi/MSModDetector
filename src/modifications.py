#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 9 2021

@author: Marjan Faizi
"""

import pyopenms
import pandas as pd
import numpy as np

class Modifications(object):
    """
    This class defines the masses of post-translational modifications (PTM).
    The input of this class should be a table with the PTM name, acronym, mass and the corresponding upper bound.
    The upper bound determines the maximal amount of a specific PTM type on a protein. 
    If the mass caused by the PTM is not given in the table it will be taken from the pyopenms class.
    """
    
    def __init__(self, modfications_file_name):
        self.modifications_table =  pd.read_csv(modfications_file_name, sep=';')
        self.modifications = self.modifications_table['PTM name'].values.tolist()
        self.acronyms = self.modifications_table['PTM acronym'].values.tolist()
        self.default_upper_bound = 20.0
        self.upper_bounds = self.modifications_table['upper bound'].replace(np.nan, self.default_upper_bound).values.tolist()
        self.ptm_masses = self.modifications_table['PTM mass'].values.tolist()

    def get_modification_masses(self, mass_type='average'):
        if mass_type == 'average':
            self.modification_masses = [self.get_average_mass(modification_type) for modification_type in self.modifications]
            
        elif mass_type == 'monoisotopic':
            self.modification_masses = [self.get_monoisotopic_mass(modification_type) for modification_type in self.modifications]


    def get_average_mass(self, modification_type):
        mod = pyopenms.ModificationsDB().getModification(modification_type)
        average_mass = mod.getDiffAverageMass()
        return average_mass


    def get_monoisotopic_mass(self, modification_type):
        mod = pyopenms.ModificationsDB().getModification(modification_type)
        monoisotopic_mass = mod.getDiffMonoMass()
        return monoisotopic_mass
