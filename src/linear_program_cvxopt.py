#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 15 2021

@author: Marjan Faizi
"""

import numpy as np
from cvxopt import matrix, glpk


class LinearProgramCVXOPT(object):
    """
    This class construct a linear program (LP) to determine a possible combination of post-translational modifications (PTM) for a given mass shift.
    The PTM pattern is determined by minimizing the number of PTMs. 
    The linear program is defined as follows: 
                                                min   1 * x 
                                                s.t.  ptm_mass_shifts * x <=  observed_mass_shift + mass_error, 
                                                     -ptm_mass_shifts * x <= -observed_mass_shift + mass_error,
                                                                     -x_i <= 0,
                                                                      x_i <= upper_bound_i,
                                                                   -1 * x <= min_number_ptms


    ptm_mass_shifts is a row vector and x a column vector containing the unknown amounts of each PTM type accounting for the total mass shift.
    The upper bound for each element in x is determined by counting the possible modification sites for a given amino acid sequence.
    """


    def __init__(self, ptm_mass_shifts, upper_bounds):
        self.ptm_mass_shifts = ptm_mass_shifts
        self.upper_bounds = upper_bounds
        self.x_values = None
        self.mass_error = 0.0
        self.min_number_ptms = 0
        self.observed_mass_shift = 0
        glpk.options['msg_lev'] = 'GLP_MSG_OFF'  


    def set_observed_mass_shift(self, observed_mass_shift):
        self.observed_mass_shift = observed_mass_shift

    		
    def set_mass_error(self, mass_error):
        self.mass_error =  mass_error
        
        
    def set_min_number_ptms(self, min_number_ptms):
        self.min_number_ptms = min_number_ptms
    
    def get_error(self, solution):
        theoretical_mass_shift = np.dot(self.ptm_mass_shifts, solution)
        error = abs(theoretical_mass_shift - self.observed_mass_shift)[0]
        return error


    def solve_linear_program(self, solver='cvxopt_glpk'):
        number_variables = len(self.ptm_mass_shifts)
        ones = np.ones(number_variables)
        inequality_lhs = np.vstack([self.ptm_mass_shifts, -self.ptm_mass_shifts, -np.identity(number_variables), np.identity(number_variables), -ones])
        A = matrix(inequality_lhs)
        lower_bounds = np.zeros(number_variables)
        inequality_rhs = np.vstack([self.observed_mass_shift+self.mass_error, -self.observed_mass_shift+self.mass_error, lower_bounds.reshape(-1,1), 
                                    self.upper_bounds.reshape(-1,1), -self.min_number_ptms])
        b = matrix(inequality_rhs)
        c = matrix(ones)
        status, solution = glpk.ilp(c, A, b, I=set(range(number_variables)))
        return status, solution

    

"""
from linear_program_cvxopt import LinearProgramCVXOPT
ptm = np.array([80.0,40.0])
ub = np.array([10,10])
lp = LinearProgramCVXOPT(ptm, ub)
lp.set_observed_mass_shift(159.4)
lp.set_mass_error(1.93)
lp.set_min_number_ptms(4)
stat, sol = lp.solve_linear_program()
print(sol)

"""
