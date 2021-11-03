# myconfig.py:

# Path to the mass spec data
path = '../data/raw_data/P04637/'

# This regular expression specifies the ending of the file names that should be read all at once
file_name_ending = '*.mzml'

# list of all replicate names as they are in the file names
replicates = ['rep1', 'rep5', 'rep6']

# list of all condition names as they are in the file names
conditions = ['nutlin_only', 'xray_2hr', 'xray_7hr', 'xray-nutlin', 'uv_7hr']

# used to determine the number of subplots
number_of_conditions = len(conditions)

# color for each condition and the respective order in the plots
color_order = [['skyblue', 0], ['yellowgreen', 1], ['lightseagreen', 2], ['chocolate', 3], ['mediumpurple', 4]]
color_palette = dict(zip(conditions, color_order))

# Name and location of the modification file 
modfication_file_name = '../data/modifications/modifications_P04637.csv'

# Set mass range to search for shifts 
mass_start_range = 43750.0
mass_end_range = 44650.0

# Theoretical average mass of the unmodified species (in Da)
unmodified_species_mass = 43653.1778

# The standard deviation of the data points within the search window determine the noise level
# The threshold of the noise level can be decreased with this parameter
noise_level_fraction = 0.5

# The fit of the gaussian distribution to the observed isotope distribution is evaluated by the chi-squared test
# A high p-value indicates a better fit; both distributions are less likely to differ from each other
pvalue_threshold = 0.1

# determine window size used to fit the gaussian distribution
# lb and ub set the percentage of peaks within the distribution that should be considered for the fit
window_size_lb = 0.5
window_size_ub = 0.95

# mass error in ppm and converted in Dalton
mass_error_ppm = 10
mass_error_Da = mass_error_ppm*1e-6*unmodified_species_mass

# Average masses within this distance should be binned together and the maximal bin size should be kept
max_bin_size = mass_error_Da

# If two peaks are within this distance (given in Da) then the lower peak is removed  
distance_threshold_adjacent_peaks = 0.6

# Set this to be true if the mass shifts should be calculated and reported in the output table
calculate_mass_shifts = True

# This mass tolerance in Da is used as default for the linear programming problem 
mass_tolerance = 1 + mass_error_Da

# Set this to be true if the PTM patterns should be determined and reported in the output table
# Only the PTM pattern with the least amount of PTMs will be selected to be displayed
determine_ptm_patterns = True
