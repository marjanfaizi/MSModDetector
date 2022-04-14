# myconfig.py:
import utils


# This regular expression specifies the ending of the file names that should be read all at once
file_names = '../data/*.mzML'

# list of all replicate names as they are in the file names
replicates = ['rep1']

# list of all condition names as they are in the file names
conditions = ['simulated']

# used to determine the number of subplots
number_of_conditions = len(conditions)

# color for each condition and the respective order in the plots
color_order = [['skyblue', 0]]
color_palette = dict(zip(conditions, color_order))

# Name and location of the modification file 
modfication_file_name = '../data/modifications/modifications_P04637.csv'

# Fasta file for protein of interest
fasta_file_name = '../data/fasta_files/P04637.fasta'

# Theoretical average mass of the unmodified species (in Da)
unmodified_species_mass, stddev_isotope_distribution = utils.mean_and_stddev_of_isotope_distribution(fasta_file_name, 100)

# Set mass range to search for shifts 
mass_start_range = 43600.0
mass_end_range = 44520.0

# The standard deviation of the data points within the search window determine the noise level
# The threshold of the noise level can be decreased with this parameter
noise_level_fraction = 0.25

# The fit of the gaussian distribution to the observed isotope distribution is evaluated by the chi-squared test
# A high p-value indicates a better fit; both distributions are less likely to differ from each other
pvalue_threshold = 0.1

# determine window size used to fit the gaussian distribution
# lb and ub set the percentage of peaks within the distribution that should be considered for the fit
window_size_lb = 0.2
window_size_ub = 0.8

# allowed overlap of window sizes that are used for fitting
allowed_overlap = 0.5

# mass error in ppm and converted in Dalton
mass_error_ppm = 20

# This mass tolerance in Da is used as default for the linear programming problem 
mass_tolerance = mass_error_ppm*1e-6*unmodified_species_mass

# Average masses within this distance should be binned together and the maximal bin size should be kept
bin_peaks = False
max_bin_size = 0

# If two peaks are within this distance (given in Da) then the lower peak is removed  
distance_threshold_adjacent_peaks = 0.6

# Solve optimization k times and report the best k optimal solutions
laps_run_lp = 10

# Choose between two objective functions: 
# 1) min_ptm: minimize total amount of PTMs on a single protein
# 2) min_err: minimize error between observed and inferred mass shift
objective_fun = "min_err"

# Set this to be true if the mass shifts should be calculated and reported in the output table
calculate_mass_shifts = True


# Set this to be true if the PTM patterns should be determined and reported in the output table
# Only the PTM pattern with the least amount of PTMs will be selected to be displayed
determine_ptm_patterns = True