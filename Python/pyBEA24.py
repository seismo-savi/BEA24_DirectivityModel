#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Time    :   2025/03/10 10:02:27
Author  :   Savvas Marcou 
Contact :   savvas.marcou@berkeley.edu
This is a Python implementation of the Bayless et al. (2024) Directivity Model: Bea24
Sourced from GitHub: https://github.com/seismo-savi/BEA24_DirectivityModel/tree/main
Python translation of original Matlab code supported by Gemini and ChatGPT
'''
import numpy as np
from scipy.interpolate import interp1d

def Bea24(M, U, T, Smax1, Smax2, Ztor, Rake, Period, Version):
    """
    Matlab function for the Bayless et al. (2024) Directivity Model: Bea24
    as described in USGS External Grants Report G22AP00199

    Jeff Bayless (jeff.bayless@aecom.com)
    Created: Feb 2024
    Copyright (c) 2024, Jeff Bayless, covered by GNU General Public License
    All rights reserved.

    Computes the directivity adjustment and phi reduction for ground motion prediction.

    Args:
        M (float): Moment magnitude, constrained to 6 <= M <= 8.
        U (numpy.ndarray): GC2 coordinate in the strike-parallel direction (km), shape (n, 1).
        T (numpy.ndarray): GC2 coordinate in the fault-normal direction (km), shape (n, 1).
            - The U coordinate must be referenced to the rupture surface trace ordinate 
            of the up-dip projection of the hypocenter.
        Smax1 (float): Maximum S in the anti-strike direction (km, negative value).
        Smax2 (float): Maximum S in the strike direction (km, positive value).
        Ztor (float): Depth to the top of rupture (km, positive value).
        Rake (float): Characteristic rupture rake angle (degrees), within:
            - [-180, -150] or [-30, 30] or [150, 180] for strike-slip ruptures.
        Period (float): Spectral period for fD and PhiRed computation (seconds), constrained to 0.01 <= Period <= 10.
        Version (int): Model version flag:
            - 1: Simulation-based
            - 2: NGA-W2 data-based

    Returns:
        tuple: A tuple containing:
        
        - fD (numpy.ndarray): Directivity adjustment in natural log units, shape (n, 1000),
        computed at 1000 log-spaced periods between 0.01 and 10 sec.
        - fDi (numpy.ndarray): Directivity adjustment at the specified `Period`, shape (n, 1).
        - PhiRed (numpy.ndarray): Phi reduction, shape (n, 1000).
        - PhiRedi (numpy.ndarray): Phi reduction at the specified `Period`, shape (n, 1).
        - PredicFuncs (dict): A dictionary containing period-independent predictors:
            - "fG" (numpy.ndarray): Uncentered geometric directivity predictor, shape (n, 1).
            - "fGprime" (numpy.ndarray): Centered geometric directivity predictor, shape (n, 1).
            - "fGbar" (numpy.ndarray): Directivity predictor centering term, shape (n, 1).
            - "fdist" (numpy.ndarray): Distance taper, shape (n, 1).
            - "fztor" (numpy.ndarray): Ztor taper, shape (n, 1).
            - "ftheta" (numpy.ndarray): Azimuthal directivity component, shape (n, 1).
            - "fs2" (numpy.ndarray): Rupture travel distance component, shape (n, 1).
            - "A" (numpy.ndarray): Period- and magnitude-dependent lower and upper bounds of fD, shape (1, 1000).
        - Other (dict): A dictionary containing additional parameters:
            - "Per" (numpy.ndarray): Periods at which fD and PhiRed are computed, shape (1, 1000).
            - "Rmax" (float): Maximum distance of the distance taper.
            - "Footprint" (numpy.ndarray): Boolean mask for sites within the directivity effect footprint, shape (n, 1).
            - "Tpeak" (float): Peak period of the directivity effect.
            - "k" (float): Logistic function slope (model coefficient).
            - "Amax" (float): Limiting upper and lower bounds of A (model coefficient).
            - "Rst" (numpy.ndarray): Distance from the surface trace (km), shape (n, 1).
            - "Ry0" (numpy.ndarray): Ry0 distance (km), shape (n, 1).
            - "S2" (numpy.ndarray): Generalized rupture travel distance parameter, shape (n, 1).
            - "theta" (numpy.ndarray): Theta angle (degrees), shape (n, 1).
    """
    # Impose limits on M
    if M > 8:
        raise ValueError("Upper M limit is 8.0")
    if M < 6:
        raise ValueError("Lower M limit is 6.0")
    
    # Determine constants based on model version
    if Version == 1:
        Amax, k, SigG = 0.54, 1.58, 0.38
        PhiPer = np.array([0.01, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
        e1 = np.array([0.00, 0.000, 0.0003, 0.011, 0.038, 0.072, 0.107, 0.143, 0.172, 0.189, 0.195, 0.206, 0.200])
    elif Version == 2:
        Amax, k, SigG = 0.34, 1.58, 0.26
        PhiPer = np.array([0.01, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
        e1 = np.array([0.00, 0.000, 0.0024, 0.0074, 0.024, 0.041, 0.064, 0.076, 0.091, 0.110, 0.124, 0.145, 0.157])
    
    Per = np.logspace(-2, 1, 1000)
    Tpeak = 10 ** (-2.15 + 0.404 * M)
    x = np.log10(Per / Tpeak)
    A = Amax * np.exp(-x**2 / (2 * SigG**2))
    e1interp = interp1d(np.log(PhiPer), e1, kind='linear', fill_value='extrapolate')(np.log(Per))
    
    # Convert U to S
    S = np.where(U < 0, -np.minimum(np.abs(U), np.abs(Smax1)), np.minimum(np.abs(U), np.abs(Smax2)))
    Ry = np.where(U >= 0, U - Smax2, np.abs(U) - np.abs(Smax1))
    Ry[(U <= Smax2) & (U >= Smax1)] = 0
    
    Srake = S * np.cos(np.radians(Rake))
    Dmin = 3
    S2 = np.sqrt(Dmin**2 + Srake**2)
    fs2 = np.log(S2)
    fs2 = np.minimum(fs2, np.log(465))
    
    theta = np.abs(np.arctan(T / U))
    theta[np.isnan(theta)] = 0
    ftheta = np.abs(np.cos(2 * theta))
    
    R = np.sqrt(T**2 + Ry**2 + Ztor**2)
    Rmax = min(80, -60 + 20 * M) if M <= 7 else 80
    Footprint = R <= Rmax
    fdist = np.where(Footprint, 1 - np.exp(-4 * Rmax / R + 4), 0)
    
    fztor = np.where(np.abs(Ztor) < 20, 1 - np.abs(Ztor) / 20, 0)
    fG = fs2 * ftheta
    fGbar = np.array([centerfunc(Smax2, np.abs(Smax1), r, Dmin, Rake) for r in R])
    fGprime = (fG - fGbar) * fdist * fztor
    
    fD = A * (2 / (1 + np.exp(-k * fGprime)) - 1)
    ti = np.argmin(np.abs(Per - Period))
    fDi = fD[:, ti]
    
    PhiRed = np.tile(e1interp, (len(fD), 1))
    PhiRed[~Footprint, :] = 0
    PhiRedi = PhiRed[:, ti]
    
    PredicFuncs = {
        'fG': fG, 'fGprime': fGprime, 'fGbar': fGbar, 'fdist': fdist,
        'fztor': fztor, 'ftheta': ftheta, 'fs2': fs2, 'A': A
    }
    
    Other = {
        'Per': Per, 'Rmax': Rmax, 'Footprint': Footprint, 'Tpeak': Tpeak,
        'k': k, 'Amax': Amax, 'Rst': R, 'Ry0': Ry, 'S2': S2, 'theta': theta
    }
    
    return fD, fDi, PhiRed, PhiRedi, PredicFuncs, Other


def centerfunc(L1, L2, R, D, Rake):
    """
    this function calculates the centering term, fGbar, for a given rupture dimension, hypocenter location, rake angle, and distance

    Args:
        L1 (float): Maximum S in the anti-strike direction (km, negative value).
        L2 (float): Maximum S in the strike direction (km, positive value).
        R (float): Distance from the surface trace (km).
        D (float): Depth to the top of rupture (km, positive value).
        Rake (float): Characteristic rupture rake angle (degrees), within:
            - [-180, -150] or [-30, 30] or [150, 180] for strike-slip ruptures.
    Returns:
        fGbar (float): Centering term for the directivity model.
    """
    if R < 0.1:
        R = 0.1

    dx = 0.1
    
    # I1 - between the ends of the fault, strike direction
    x = np.arange(0, L1 + dx, dx)
    xr = x * np.cos(np.radians(Rake))
    s = np.sqrt(xr**2 + D**2)
    y1 = np.log(s) * np.abs(np.cos(2 * np.arctan(R / x)))
    
    # I2 - between the ends of the fault, anti-strike direction
    x = np.arange(0, L2 + dx, dx)
    xr = x * np.cos(np.radians(Rake))
    s = np.sqrt(xr**2 + D**2)
    y2 = np.log(s) * np.abs(np.cos(2 * np.arctan(R / x)))
    
    # I3 - off the end of the fault, strike direction
    l = L1
    lr = l * np.cos(np.radians(Rake))
    s = np.sqrt(lr**2 + D**2)
    x = np.arange(l + dx, l + R + dx, dx)
    r = np.sqrt(R**2 - (x - l)**2)
    y3 = np.log(s) * np.abs(np.cos(2 * np.arctan(r / x)))
    
    # I4 - off the end, anti-strike direction
    l = L2
    lr = l * np.cos(np.radians(Rake))
    s = np.sqrt(lr**2 + D**2)
    x = np.arange(l + dx, l + R + dx, dx)
    r = np.sqrt(R**2 - (x - l)**2)
    y4 = np.log(s) * np.abs(np.cos(2 * np.arctan(r / x)))
    
    # Total
    fGbar = np.mean(np.concatenate([y1, y2, y3, y4]))

    return fGbar
