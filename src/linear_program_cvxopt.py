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
    The PTM pattern is determined either by (1) minimizing the number of PTMs or by (2) minimizing the error between observed and inferred mass shift. 
    
    (1) The first linear program is defined as follows: 
                                                min   1 * x 
                                                s.t.  ptm_mass_shifts * x <=  observed_mass_shift + max_mass_error, 
                                                     -ptm_mass_shifts * x <= -observed_mass_shift + max_mass_error,
                                                                     -x_i <=  0,
                                                                      x_i <=  upper_bound_i,
                                                                   -1 * x <= -min_number_ptms

    ptm_mass_shifts is a row vector and x a column vector containing the unknown amounts of each PTM type accounting for the total mass shift.
    The upper bound for each element in x is determined by counting the possible modification sites for a given amino acid sequence.
   
    (2) The second linear program minimizes the difference between theoretical and observed mass shift:
                                                min   | ptm_mass_shift * x - observed_mass_shift |
                                                s.t.                 -x_i <=  0,
                                                                      x_i <=  upper_bound_i                                                                
                                                                   
    This problem is equivalent to:
                                                min   error
                                                s.t.  ptm_mass_shifts * x - error <=  observed_mass_shift,
                                                     -ptm_mass_shifts * x + error <= -observed_mass_shift,
                                                                             -x_i <=  0,
                                                                           -error <= -min_error,   
                                                                              x_i <=  upper_bound_i,
                                                                            error <=  max_mass_error                                                                            
    """


    def __init__(self, ptm_mass_shifts, upper_bounds):
        self.ptm_mass_shifts = ptm_mass_shifts
        self.upper_bounds = upper_bounds
        self.max_mass_error = 0.0
        self.observed_mass_shift = 0
        glpk.options['msg_lev'] = 'GLP_MSG_OFF'  


    def set_observed_mass_shift(self, observed_mass_shift):
        self.observed_mass_shift = observed_mass_shift

    		
    def set_max_mass_error(self, max_mass_error):
        self.max_mass_error =  max_mass_error


    def get_error(self, solution):
        theoretical_mass_shift = np.dot(self.ptm_mass_shifts, solution)
        error = abs(theoretical_mass_shift - self.observed_mass_shift)[0]
        return error


    def solve_lp_min_ptms(self, min_number_ptms):
        number_variables = len(self.ptm_mass_shifts)
        ones = np.ones(number_variables)
        inequality_lhs = np.vstack([self.ptm_mass_shifts-self.observed_mass_shift, -self.ptm_mass_shifts+self.observed_mass_shift, -np.identity(number_variables), np.identity(number_variables), -ones])
        A = matrix(inequality_lhs)
        lower_bounds = np.zeros(number_variables)
        inequality_rhs = np.vstack([self.max_mass_error, self.max_mass_error, lower_bounds.reshape(-1,1), 
                                    self.upper_bounds.reshape(-1,1), -min_number_ptms])
        b = matrix(inequality_rhs)
        c = matrix(ones)
        status, solution = glpk.ilp(c, A, b, I=set(range(number_variables)))
        return status, solution


    def solve_lp_min_error(self, min_error, number_ptms=None):
        number_variables = len(self.ptm_mass_shifts)
        ones = np.ones(number_variables)
        zeros = np.zeros(number_variables) 
        lower_bounds = zeros.reshape(-1,1)
        
        if number_ptms != None:
            min_number_ptms = -number_ptms
            max_number_ptms = number_ptms
        else:
            min_number_ptms = 0
            max_number_ptms = 20
        
        # mininimize  ptm_mass_shifts * x - observed_mass_shift, if ptm_mass_shifts * x >= observed_mass_shift
        inequality_lhs = np.vstack([np.hstack([self.ptm_mass_shifts, -1]), np.hstack([-self.ptm_mass_shifts, 1]),                                     
                                    -np.identity(number_variables+1), np.identity(number_variables+1),
                                    np.hstack([-self.ptm_mass_shifts, 0]), np.hstack([-ones, 0]), np.hstack([ones, 0])])         
        A = matrix(inequality_lhs)
        inequality_rhs = np.vstack([self.max_mass_error, -min_error,  lower_bounds, self.observed_mass_shift, self.upper_bounds.reshape(-1,1),
                                    -self.observed_mass_shift, self.observed_mass_shift, min_number_ptms, max_number_ptms])
        b = matrix(inequality_rhs)
        c = matrix(np.hstack([self.ptm_mass_shifts,-1]))
        status_min, solution_min = glpk.ilp(c, A, b, I=set(range(number_variables)))
        
        # maximize ptm_mass_shifts * x - observed_mass_shift, if ptm_mass_shifts * x <= observed_mass_shift
        inequality_lhs = np.vstack([np.hstack([self.ptm_mass_shifts, -1]), np.hstack([-self.ptm_mass_shifts, 1]),                                     
                                    -np.identity(number_variables+1), np.identity(number_variables+1),
                                    np.hstack([self.ptm_mass_shifts, 0]), np.hstack([-ones, 0]), np.hstack([ones, 0])])         
        A = matrix(inequality_lhs)
        inequality_rhs = np.vstack([self.max_mass_error, -min_error,  lower_bounds, self.observed_mass_shift, self.upper_bounds.reshape(-1,1),
                                    self.observed_mass_shift, self.observed_mass_shift, min_number_ptms, max_number_ptms])
        b = matrix(inequality_rhs)
        c = matrix(np.hstack([-self.ptm_mass_shifts,1]))
        status_max, solution_max = glpk.ilp(c, A, b, I=set(range(number_variables)))
        
        if solution_min and solution_max:
            error_min = self.get_error(solution_min[:-1]) 
            error_max = self.get_error(solution_max[:-1]) 
            if error_min <= error_max:
                return status_min, solution_min
            else:
                return status_max, solution_max
    
        elif solution_min and not solution_max:
            return status_min, solution_min
        
        else:
            return status_max, solution_max
    

    """
    def solve_lp_min_error(self, min_error, number_ptms=None):
        number_variables = len(self.ptm_mass_shifts)
        ones = np.ones(number_variables)
        zeros = np.zeros(number_variables) 
        lower_bounds = zeros.reshape(-1,1)
        
        if number_ptms != None:
            min_number_ptms = -number_ptms
            max_number_ptms = number_ptms
        else:
            min_number_ptms = 0
            max_number_ptms = 20
        
        inequality_lhs = np.vstack([np.hstack([self.ptm_mass_shifts, -1]), np.hstack([-self.ptm_mass_shifts, 1]),                                     
                                    -np.identity(number_variables+1), np.identity(number_variables+1), 
                                    np.hstack([-ones, 0]), np.hstack([ones, 0])])      
        A = matrix(inequality_lhs)
        inequality_rhs = np.vstack([self.observed_mass_shift, -self.observed_mass_shift, lower_bounds, -min_error, 
                                    self.upper_bounds.reshape(-1,1), self.max_mass_error, min_number_ptms, max_number_ptms])
        b = matrix(inequality_rhs)
        c = matrix(np.hstack([zeros,1]))
        status, solution = glpk.ilp(c, A, b, I=set(range(number_variables)))
        return status, solution
    """