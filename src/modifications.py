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
    This class defines the masses of post-translational modifications (PTM) and a mapping from PTMs to amino acids.
    The PTM to amino acid mapping represents the biological fact that specific PTMs only bind to selected amino acid residues.   
    
    Source for PTM to amino acid mapping: 
        - Hafner et al. The multiple mechanisms that regulate p53 activity and cell fate (2019)
        - DeHart et al. Extensive Post-translational Modification of Active and Inactivated Forms of Endogenous p53 (2014)
        
    """

    
    def __init__(self, modfication_file_name):
        self.modification_table =  pd.read_csv(modfication_file_name, sep=';')
        self.modifications = self.modification_table['PTM id'].values.tolist()
        self.acronyms = self.modification_table['PTM acronym'].values.tolist()
        self.default_upper_bound = 20.0
        self.upper_bounds = self.modification_table['upper bound'].replace(np.nan, self.default_upper_bound).values.tolist()
        self.modification_masses = 0.0

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
   
