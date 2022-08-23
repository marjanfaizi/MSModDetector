# config.py:

    
# This regular expression specifies the ending of the file names that should be read all at once
file_names = "raw_data/*.mzml"

# list of all replicate names as they are in the file names
replicates = ["rep5", "rep6"] # rep1, rep5, rep6, rep9

# list of all condition names as they are in the file names
#conditions = ["nutlin_only", "xray_2hr", "xray_7hr", "xray-nutlin", "uv_7hr"]
conditions = ["nutlin_only", "uv_7hr"]

# used to determine the number of subplots
number_of_conditions = len(conditions)

# color for each condition and the respective order in the plots
#color_order = [["skyblue", 0], ["yellowgreen", 1], ["lightseagreen", 2], ["chocolate", 3], ["mediumpurple", 4]]
color_order = [["skyblue", 0], ["mediumpurple", 1]]
color_palette = dict(zip(conditions, color_order))

# Name and location of the modification file 
modfication_file_name = "modifications/modifications_P04637.csv"

# Fasta file for protein of interest
fasta_file_name = "fasta_files/P04637.fasta"

# Set mass range to search for shifts 
mass_range_start = 43750.0
mass_range_end = 44520.0

# The standard deviation of the data points within the search window determine the noise level
# The threshold of the noise level can be decreased with this parameter
noise_level_fraction = 0.5

# The fit of the gaussian distribution to the observed isotope distribution is evaluated by the chi-squared test
# A high p-value indicates a better fit; both distributions are less likely to differ from each other
pvalue_threshold = 0.99

# set window size used to fit the gaussian distribution
window_size = 9

# mass error in ppm and converted in Dalton
mass_error_ppm = 30

# Average masses within this distance should be binned together and the maximal bin size should be kept
bin_peaks = True

# If two peaks are within this distance (given in Da) then the lower peak is removed  
distance_threshold_adjacent_peaks = 0.6

# Solve optimization k times and report the best k optimal solutions
laps_run_lp = 10

# Choose between two objective functions: 
# 1) min_ptm: minimize total amount of PTMs on a single protein
# 2) min_err: minimize error between observed and inferred mass shift
# 3) min_both: minimize error and total amount of PTMs
objective_fun = "min_ptm"

# Set this to be true if the mass shifts should be calculated and reported in the output table
calculate_mass_shifts = True


# Set this to be true if the PTM patterns should be determined and reported in the output table
# Only the PTM pattern with the least amount of PTMs will be selected to be displayed
determine_ptm_patterns = True
