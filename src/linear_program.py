#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 2021

@author: Marjan Faizi
"""
import numpy as np
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpDot, GLPK
import sys

class LinearProgram(object):
    """
    This class construct a linear program (LP) to determine a possible combination of post-translational modifications (PTM) for a given mass shift.
    The PTM pattern is determined by minimizing the absolute difference between the observed and the theoretical mass shift. 
    The linear program is defined as follows: 
                                                min |theoretical_mass_shift - observed_mass_shift|
                                                s.t. ptm_mass_shifts * x = theoretical_mass_shift , 
                                                                        0 <= x_i <= upper_bound_i ,
                                                   
    ptm_mass_shifts is a row vector and x a column vector containing the unknown amounts of each PTM type accounting for the total mass shift.
    The upper bound for each element in x is determined by counting the possible modification sites for a given amino acid sequence.

    Optimization problem of an absolute value has to be reformulated as follows: 
             i) if theoretical_mass_shift >= observed_mass_shift, then minimize (theoretical_mass_shift - observed_mass_shift)
             ii) if theoretical_mass_shift < observed_mass_shift, then maximize (theoretical_mass_shift - observed_mass_shift)
    """
    
    def __init__(self, lp_model_name='lp_model'):
        self.lp_model_name = lp_model_name
        self.objective_value = -1.0
        self.x_values = None
        self.minimal_mass_error = -1.0 


    def set_observed_mass_shift(self, observed_mass_shift=None):
        self.observed_mass_shift = observed_mass_shift


    def set_ptm_mass_shifts(self, ptm_mass_shifts):
        if type(ptm_mass_shifts) == list:
            self.ptm_mass_shifts = np.array(ptm_mass_shifts) 
        else:
            self.ptm_mass_shifts = ptm_mass_shifts
    
		
    def set_minimal_mass_error(self, minimal_mass_error):
        self.minimal_mass_error = minimal_mass_error

        
    def get_objective_value(self):
        return self.objective_value

    
    def get_x_values(self):
        return self.x_values


    def reset_mass_error(self):
        self.minimal_mass_error = -1.0 
        self.objective_value = -1.0


    def solve_linear_program_for_objective_with_absolute_value(self):                       
        # throw exception if observed mass was not set before calling this function 
        try:
            lp_model_min, x_min = self.__minimize_if_objective_is_positive()
            lp_model_max, x_max = self.__maximize_if_objective_is_negative()
        except: 
            print('ERROR: observed mass shift has to be set before running the linear program.')
            sys.exit()
        
        self.__solve_linear_program(lp_model_min)
        self.__solve_linear_program(lp_model_max)

        if (np.abs(lp_model_min.objective.value()) < np.abs(lp_model_max.objective.value())):
            self.objective_value = lp_model_min.objective.value()
            self.x_values = [xi.value() for xi in x_min]
        else:
            self.objective_value = lp_model_max.objective.value()
            self.x_values = [xi.value() for xi in x_max]
  
    
    def __calculate_theoretical_mass_shift(self, x):
        theoretical_mass_shift = lpDot(self.ptm_mass_shifts, np.array(x))
        return theoretical_mass_shift
  
    
    def __minimize_if_objective_is_positive(self):
        lp_model_min = LpProblem(name=self.lp_model_name+'_min', sense=LpMinimize)
        x_min = self.__define_variable_vector_x()
        theoretical_mass_shift = self.__calculate_theoretical_mass_shift(x_min)
        lp_model_min += (theoretical_mass_shift >= self.observed_mass_shift)
        mass_error = self.__set_objective(theoretical_mass_shift) 
        lp_model_min += (mass_error >= self.minimal_mass_error + 1e-3)
        lp_model_min += mass_error
        return lp_model_min, x_min
    
    
    def __maximize_if_objective_is_negative(self):
        lp_model_max = LpProblem(name=self.lp_model_name+'_max', sense=LpMaximize)
        x_max = self.__define_variable_vector_x()
        theoretical_mass_shift = self.__calculate_theoretical_mass_shift(x_max)
        lp_model_max += (theoretical_mass_shift <= self.observed_mass_shift)
        mass_error = self.__set_objective(theoretical_mass_shift) 
        lp_model_max += (mass_error <= -(self.minimal_mass_error + 1e-3))
        lp_model_max += mass_error
        return lp_model_max, x_max
    
    
    def __define_variable_vector_x(self):
        number_ptms = len(self.ptm_mass_shifts)
        x = [LpVariable(name=f'x{i}', lowBound=0, upBound=self.upper_bounds[i], cat="Integer") for i in range(number_ptms)]
        return x
  
    
    def set_upper_bounds(self, upper_bounds):
        self.upper_bounds = upper_bounds
        
    
    def __set_objective(self, theoretical_mass_shift):
        mass_error = theoretical_mass_shift - self.observed_mass_shift
        return mass_error
  
    
    def __solve_linear_program(self, lp_model):
        solver_status = lp_model.solve(solver=GLPK(msg=False))
        return solver_status
    

