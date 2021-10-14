# myconfig.py:

# Path to the mass spec data
path = '../data/'

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
modfication_file_name = '../data/modifications_P04637.csv'

# Set the maximal mass shift to consider (in Da)
max_mass_shift = 800.0

# Set the start of the search window within the mass spectrum (in Da)
start_mass_range = 43750.0 #35837.0 #41080.0 

# Initial guess for the mass of the unmodified species (in Da)
unmodified_species_mass_init = 43770.0 #41100.0 #35857.0

# Search for the mass of the unmodified species within this range -/+ mass_tol (in Da)
unmodified_species_mass_tol = 5.0 

# The fit of the gaussian distribution to the observed isotope distribution is evaluated by the chi-squared test
# A high p-value indicates a better fit; both distributions are less likely to differ from each other
pvalue_threshold = 0.05

# determine window size used to fit the gaussian distribution
# lb and ub set the percentage of peaks within the distribution that should be considered for the fit
window_size_lb = 0.5
window_size_ub = 0.95

# difference between idetnified masses that should be in one bin
bin_size_mass_shifts = 2.5

# If two peaks are within this distance (given in Da) then the lower peak is removed  
distance_threshold_adjacent_peaks = 0.6

# Set this to be true if the mass shifts should be calculated and reported in the output table
calculate_mass_shifts = True

# This mass error in Da is used as default for the linear programming problem 
mass_error = bin_size_mass_shifts/2

# Set this to be true if the PTM patterns should be determined and reported in the output table
# Only the PTM pattern with the least amount of PTMs will be selected to be displayed
determine_ptm_patterns = True
