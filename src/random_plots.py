# -*- coding: utf-8 -*-
"""
Created on Oct 20 2021

@author: Marjan Faizi
"""

import glob
import matplotlib.pyplot as plt
import seaborn as sns

from mass_spec_data import MassSpecData
import myconfig as config


plt.style.use("seaborn-pastel")

condition = "nutlin_only"
file_names = [file for file in glob.glob("../data/"+condition+"*.mzml")] 
replicates = ["rep1", "rep5", "rep6"]
max_mass_shift = 800


for ix, name in enumerate(file_names):
    data = MassSpecData(name)
    data.set_search_window_mass_range(config.start_mass_range, max_mass_shift)        

    max_intensity = data.intensities.max()
    data.convert_to_relative_intensities(max_intensity)
    rescaling_factor = max_intensity/100.0

    all_peaks = data.picking_peaks()
    trimmed_peaks = data.remove_adjacent_peaks(all_peaks, config.distance_threshold_adjacent_peaks)   
    trimmed_peaks_in_search_window = data.determine_search_window(trimmed_peaks)
    
    sn_threshold = config.noise_level_fraction*trimmed_peaks_in_search_window[:,1].std()

    if replicates[ix] == "rep5":
        plt.plot(data.masses, data.intensities, "-", color="0.5")
        plt.plot(all_peaks[:,0], all_peaks[:,1], "r.")
        plt.plot(trimmed_peaks_in_search_window[:,0], trimmed_peaks_in_search_window[:,1], "g.")
        plt.xlim(data.search_window_start_mass-10, data.search_window_end_mass+10)
        plt.axhline(y=sn_threshold, c='0.1', lw=0.3)
        plt.axhline(y=2*sn_threshold, c='0.3', lw=0.3)
        plt.show()
        
        
        
        
        
        
    print('\nCreate plots.')
    
    means = mass_shifts.identified_masses_df[mass_shifts.avg_mass_col_name].values
    for rep in config.replicates:
        output_fig = plt.figure(figsize=(14,7))
        gs = output_fig.add_gridspec(config.number_of_conditions, hspace=0)
        axes = gs.subplots(sharex=True, sharey=True)
        ylim_max = mass_shifts.identified_masses_df.filter(regex=mass_shifts.intensity_col_name+".*"+rep).max().max()
        
        file_names_same_replicate = [file for file in file_names if re.search(rep, file)]
        
        for cond in config.conditions:
        
            sample_name = [file for file in file_names_same_replicate if re.search(cond, file)]    
           
            if not sample_name:
                print('\nCondition not available.\n')
                sys.exit()
            else:
                sample_name = sample_name[0]
    
            data = MassSpecData(sample_name)
            data.set_search_window_mass_range(config.start_mass_range, config.max_mass_shift) 
            
            color_of_sample = config.color_palette[cond][0]
            order_in_plot = config.color_palette[cond][1]
            calibration_number = calibration_number_dict[cond+"_"+rep]
            axes[order_in_plot].plot(data.masses+calibration_number, data.intensities, label=cond, color=color_of_sample)
    
            masses = mass_shifts.identified_masses_df[mass_shifts.mass_col_name+cond+"_"+rep].dropna().values
            intensities = mass_shifts.identified_masses_df[mass_shifts.intensity_col_name+cond+"_"+rep].dropna().values
            stddevs = utils.mapping_mass_to_stddev(masses)
            x_gauss_func = np.arange(data.search_window_start_mass, data.search_window_end_mass)
            y_gauss_func = utils.multi_gaussian(x_gauss_func, intensities, masses, stddevs)
            axes[order_in_plot].plot(x_gauss_func, y_gauss_func, color='0.3')
            axes[order_in_plot].plot(masses, intensities, '.', color='0.3')
            axes[order_in_plot].axhline(y=sn_threshold_dict[cond+"_"+rep], c='r', lw=0.3)
            axes[order_in_plot].set_xlim((data.search_window_start_mass, data.search_window_end_mass))
            axes[order_in_plot].set_ylim((-10, ylim_max*1.1))
            axes[order_in_plot].yaxis.grid()
            axes[order_in_plot].legend(fontsize=11, loc='upper right')
            for m in means:
                axes[order_in_plot].axvline(x=m, c='0.3', ls='--', lw=0.3, zorder=0)
            axes[order_in_plot].label_outer()
        
        plt.xlabel('mass (Da)'); plt.ylabel('intensity')
        output_fig.tight_layout()
        plt.savefig('../output/identified_masses_'+rep+'.pdf', dpi=800)
        plt.show()




    def keep_reproduced_results(self):
        keep_masses_indices = np.arange(len(self.identified_masses_df))
        for cond in config.conditions:
            missing_masses_count = self.identified_masses_df.filter(regex="masses .*"+cond).isnull().sum(axis=1)
            keep_masses_indices = np.intersect1d(keep_masses_indices, self.identified_masses_df[missing_masses_count>=2].index)
            
        self.identified_masses_df.drop(index=keep_masses_indices, inplace=True)
