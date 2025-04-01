#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Time    :   2025/03/10 10:51:31
Author  :   Savvas Marcou 
Contact :   savvas.marcou@berkeley.edu
'''


import numpy as np
import matplotlib.pyplot as plt

# Local imports
from pyGC2 import GC2
from pyBEA24 import Bea24

# Clear all variables and close all figures (equivalent to clear all, close all, clc in MATLAB)
# In Python, we don't need a clear command, but we can ensure the environment is clean.

# Define the grid of stations X and Y, in km
SiteX = np.arange(-80., 181., 1.)
SiteY = np.arange(-80., 181., 1.)

# Define the rupture (ftraces in MATLAB)
# Here we define the rupture as required for GC2 (Spudich and Chiou, 2015)
# ftraces is a list of dictionaries with length equal to the number of fault strands.
# Each strand contains the trace, strike, and lengths of the segments.

# Example 1: A single strand with a single segment 80 km in length
ftraces = [{
    'trace': np.array([[0., 0.], [0., 80.]]),
    'strike': np.array([0.]),
    'l': np.array([80.])
}]

# You could define other options similarly (Option 2, Option 3) if needed

nt = len(ftraces)

# Moment magnitude
M = 7.2

# Model version: 1->simulation-based, 2->NGA-W2 data-based
Version = 1

# Select the period at which to show the effect
Tdo = 3.

# Characteristic rupture parameters
Rake = 0.  # rake in deg
Ztor = 0.  # Ztor, must be positive, in km

# Specify the coordinates of the epicenter and GC2 origin, po
type = {
    'epi': np.array([0., 10.]),  # X, Y
    'po': np.array([0., 10.])    # in this case, the same as the epicenter
}

# Call the Spudich and Chiou (2015) GC2 function
type['str'] = 'JB'
discordant = False
gridflag = True

# Assuming the `GC2` function is defined elsewhere and returns the needed values
T, U, W, reference_axis, p_origin, nominal_strike, Upo = GC2(ftraces, SiteX, SiteY, type, discordant, gridflag)

# Calculate the maximum value of S in each direction for this hypocenter
# This is U calculated at the nominal strike ends
_, Uend, _, _, _, _, _, _ = GC2(ftraces, nominal_strike['a'][0, 0], nominal_strike['a'][0, 1], type, discordant, gridflag)
_, Uend2, _, _, _, _, _, _ = GC2(ftraces, nominal_strike['a'][1, 0], nominal_strike['a'][1, 1], type, discordant, gridflag)

Smax1 = min(Uend, Uend2)
Smax2 = max(Uend, Uend2)

