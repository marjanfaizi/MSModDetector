#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 5 2022

@author: Marjan Faizi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.mass_spec_data import MassSpecData
from src.gaussian_model import GaussianModel
from src import config
from src import utils

### set figure style
sns.set()
sns.set_style("white")
sns.set_context("paper")


### required input
sample_name = "../../raw_data/nutlin_only_rep5.mzml"
cond = "nutlin_only"
mass_range_start = 43755
mass_range_end = 43800
mass_range = np.arange(mass_range_start,mass_range_end)

### process data
data = MassSpecData()
data.add_raw_spectrum(sample_name)
data.set_mass_range_of_interest(config.mass_range_start, config.mass_range_end)
all_peaks = data.picking_peaks()
peaks_normalized = data.preprocess_peaks(all_peaks, config.distance_threshold_adjacent_peaks)
noise_level = config.noise_level_fraction*peaks_normalized[:,1].std()
peaks_normalized_above_noise = peaks_normalized[peaks_normalized[:,1]>noise_level]

gaussian_model = GaussianModel(cond, config.stddev_isotope_distribution)
gaussian_model.determine_variable_window_sizes(config.unmodified_species_mass, config.window_size_lb, config.window_size_ub)
gaussian_model.fit_gaussian_within_window(peaks_normalized_above_noise, config.pvalue_threshold)      

mean = gaussian_model.fitting_results["mean"]
amplitude = gaussian_model.fitting_results["amplitude"]


### figure A: raw spectrum
plt.figure(figsize=(2.,1.6))
plt.plot(data.raw_spectrum[:,0], data.raw_spectrum[:, 1], '-', color="0.3", linewidth=1)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)")
plt.ylabel("intensity (a.u.)")
plt.xlim((mass_range_start,mass_range_end))
plt.ylim((-10,800))
plt.tight_layout()
sns.despine()
plt.show()


### figure B: preprocessed spectrum, with only peaks showing
plt.figure(figsize=(2.,1.6))
plt.plot(peaks_normalized[:,0], peaks_normalized[:,1], '.', color="0.6")
plt.plot(peaks_normalized_above_noise[:,0], peaks_normalized_above_noise[:,1], '.', color="0.3")
plt.axhline(y=noise_level, c="b", lw=0.8)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)")
plt.ylabel("rel. intensity")
plt.xlim((mass_range_start,mass_range_end))
plt.tight_layout()
sns.despine()
plt.show()


### figure C: Gaussian fitted to the data
plt.figure(figsize=(2.,1.6))
plt.plot(peaks_normalized[:,0], peaks_normalized[:,1], '.', color="0.6")
plt.plot(peaks_normalized_above_noise[:,0], peaks_normalized_above_noise[:,1], '.', color="0.3")
plt.plot(mass_range, utils.gaussian(mass_range, amplitude[0], mean[0], gaussian_model.stddev), "r-", lw=0.95)
plt.plot(mass_range, utils.gaussian(mass_range, amplitude[1], mean[1], gaussian_model.stddev), "r-", lw=0.95)
plt.axhline(y=noise_level, c='b', lw=0.5)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)")
plt.ylabel("rel. intensity")
plt.xlim((mass_range_start,mass_range_end))
plt.tight_layout()
sns.despine()
plt.show()


### figure D, F: displaying mass shift with vertical lines 
plt.figure(figsize=(2.,1.6))
plt.plot(peaks_normalized[:,0], peaks_normalized[:,1], '.', color="0.6")
plt.plot(peaks_normalized_above_noise[:,0], peaks_normalized_above_noise[:,1], '.', color="0.3")
plt.plot(mass_range, utils.gaussian(mass_range, amplitude[0], mean[0], gaussian_model.stddev), "r-", lw=0.95)
plt.plot(mass_range, utils.gaussian(mass_range, amplitude[1], mean[1], gaussian_model.stddev), "r-", lw=0.95)
plt.axvline(x=mean[0], c="k", lw=0.5, ls="--")
plt.axvline(x=mean[1], c="k", lw=0.5, ls="--")
plt.plot(mean[0], amplitude[0], "r.")
plt.plot(mean[1], amplitude[1], "r.")
plt.axhline(y=noise_level, c='b', lw=0.5)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)")
plt.ylabel("rel. intensity")
plt.xlim((mass_range_start,mass_range_end))
plt.tight_layout()
sns.despine()
plt.show()


### figure E: quantifying mass shift by area under the curve
gaussian_model.refit_results(peaks_normalized, noise_level, refit_mean=True)
mean = gaussian_model.fitting_results["mean"]
amplitude = gaussian_model.fitting_results["amplitude"]

plt.figure(figsize=(2.,1.6))
plt.plot(peaks_normalized[:,0], peaks_normalized[:,1], '.', color="0.6")
plt.plot(peaks_normalized_above_noise[:,0], peaks_normalized_above_noise[:,1], '.', color="0.3")
plt.plot(mass_range, utils.gaussian(mass_range, amplitude[0], mean[0], gaussian_model.stddev), "orange", lw=0.9)
plt.plot(mass_range, utils.gaussian(mass_range, amplitude[1], mean[1], gaussian_model.stddev), "plum", lw=0.9)
plt.fill_between(mass_range, utils.gaussian(mass_range, amplitude[0], mean[0], gaussian_model.stddev), color="none", hatch="//////", edgecolor="orange")
plt.fill_between(mass_range, utils.gaussian(mass_range, amplitude[1], mean[1], gaussian_model.stddev), color="none", hatch="//////", edgecolor="plum")
plt.plot(mass_range, utils.multi_gaussian(mass_range, amplitude, mean, gaussian_model.stddev), "r-")
plt.axhline(y=noise_level, c='b', lw=0.5)
plt.xticks([])
plt.yticks([])
plt.xlabel("mass (Da)")
plt.ylabel("rel. intensity")
plt.xlim((mass_range_start,mass_range_end))
plt.tight_layout()
sns.despine()
plt.show()  




