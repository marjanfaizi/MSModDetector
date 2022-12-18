#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun 18 2022

@author: Marjan Faizi
"""


### required input parameters used to analyze the experimental p53 data

modfication_file_name = "modifications_P04637.csv"

fasta_file_name = "P04637.fasta"

mass_range_start = 43750.0

mass_range_end = 44520.0

noise_level_fraction = 0.5

window_size = 12

pvalue_threshold = 0.99999

allowed_overlap = 0.3

# list of all replicate names as they are in the file names
replicates = ["rep5"]

# list of all condition names as they are in the file names
conditions = ["nutlin_only", "uv_7hr"]

# used to determine the number of subplots
number_of_conditions = len(conditions)

# color for each condition and the respective order in the plots
#color_order = [["skyblue", 0], ["yellowgreen", 1], ["lightseagreen", 2], ["chocolate", 3], ["mediumpurple", 4]]
color_order = [["skyblue", 0],  ["yellowgreen", 1], ["lightseagreen", 2]]
color_palette = dict(zip(conditions, color_order))





