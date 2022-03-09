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
    The input of this class should be a table with the PTM id/acronym, PTM name, mass and the corresponding upper bound.
    The upper bound determines the maximal amount of a specific PTM type on a protein. 
    If the mass caused by the PTM is not given in the table it will be taken from the pyopenms class.
    """
    
    def __init__(self, modfications_file_name, aa_sequence_str):
        self.modifications_table =  pd.read_csv(modfications_file_name, sep=';')
        self.modifications = self.modifications_table["ptm_name"].values.tolist()
        self.ptm_ids = self.modifications_table["ptm_id"].values.tolist()
        self.upper_bounds = self.set_upper_bounds(aa_sequence_str)
        self.ptm_masses = self.modifications_table["ptm_mass"].values.tolist()


    def get_modification_masses(self, mass_type="average"):
        if mass_type == "average":
            self.modification_masses = [self.get_average_mass(modification_type) for modification_type in self.modifications]
            
        elif mass_type == "monoisotopic":
            self.modification_masses = [self.get_monoisotopic_mass(modification_type) for modification_type in self.modifications]


    def get_average_mass(self, modification_type):
        mod = pyopenms.ModificationsDB().getModification(modification_type)
        average_mass = mod.getDiffAverageMass()
        return average_mass


    def get_monoisotopic_mass(self, modification_type):
        mod = pyopenms.ModificationsDB().getModification(modification_type)
        monoisotopic_mass = mod.getDiffMonoMass()
        return monoisotopic_mass


    def set_upper_bounds(self, aa_sequence_str):
        default_upper_bound = 20
        self.modifications_table["upper_bound"].replace(np.nan, default_upper_bound, inplace=True)    
        for index, row in self.modifications_table.iterrows():
            sites = row["site"].split(",")
            aa_occurence_in_protein = np.array([ aa_sequence_str.count(aa)for aa in sites ]).sum() 
            if row["upper_bound"] > aa_occurence_in_protein:
                self.modifications_table.loc[index, "upper_bound"] = aa_occurence_in_protein
              
        upper_bounds = self.modifications_table["upper_bound"].values.tolist()  
        return upper_bounds
        

    
    
    
    
    
    