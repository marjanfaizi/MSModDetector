#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 2021

@author: Marjan Faizi
"""

from brainpy import isotopic_variants
from matplotlib import pyplot as plt
import numpy as np



# Generate theoretical isotopic pattern
peptide = {'H': 230, 'C': 440, 'O': 200, 'N': 100}
theoretical_isotopic_cluster = isotopic_variants(peptide, npeaks=50, charge=1)
for peak in theoretical_isotopic_cluster:
    print(peak.mz, peak.intensity)



# produce a theoretical profile using a gaussian peak shape

mz_grid = np.arange(theoretical_isotopic_cluster[0].mz - 1,
                    theoretical_isotopic_cluster[-1].mz + 1, 0.2)
intensity = np.zeros_like(mz_grid)
sigma = 0.02
for peak in theoretical_isotopic_cluster:
    # Add gaussian peak shape centered around each theoretical peak
    intensity += peak.intensity * np.exp(-(mz_grid - peak.mz) ** 2 / (2 * sigma)
            ) / (np.sqrt(2 * np.pi) * sigma)

# Normalize profile to 0-100
intensity = (intensity / intensity.max()) * 100

# Plot the isotopic distribution
plt.plot(mz_grid, intensity)
plt.xlabel("mass (Da)")
plt.ylabel("relative intensity (%)")