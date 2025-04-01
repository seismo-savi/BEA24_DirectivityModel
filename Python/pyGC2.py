#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Time    :   2025/03/10 10:02:10
Author  :   Savvas Marcou 
Contact :   savvas.marcou@berkeley.edu
Translation of the GC2.m Matlab code to Python
Sourced from GitHub: https://github.com/seismo-savi/BEA24_DirectivityModel/tree/main
Python translation of original Matlab code supported by Gemini and ChatGPT
'''

import numpy as np


# Step 1, in Brian's code was to convert origin-strike-length to fault trace coordinates; 
# this is the input to the main function above and is therefore skipped
# Step 2, Compute GC2 Nominal Strike
def comp_nominal_strike(ftraces):
    """
    Step 2: Compute GC2 Nominal Strike.

    This function calculates the nominal strike and other parameters for use later in the function link_traces. 
    The traces are in arbitrary order and are not necessarily in the direction of strike.

    Args:
        ftraces (list of dict): A list of rupture strands, each with 'trace' (ndarray) and 'strike' (ndarray).

    Returns:
        dict: A dictionary containing the calculated nominal strike parameters:
            - 'a' (ndarray): Trial vector-a, formed by connecting the two endpoints most distant from each other.
            - 'b' (ndarray): The corrected vector-b.
            - 'e' (ndarray): The directional dot product for each trace.
            - 'E' (float): The sum of the 'e' values.
    """

    # Create a matrix of trace ends, two rows per trace        
    m = len(ftraces)
    trace_ends = np.zeros((m*2, 2))
    
    for jj in range(m):
        nseg = len(ftraces[jj]['strike'])
        trace_ends[(jj)*2:(jj+1)*2, :] = ftraces[jj]['trace'][[0, nseg], :]

    # Find the two endpoints most distant from each other
    n = trace_ends.shape[0]
    maxdist = -1
    for ii in range(n-1):
        for kk in range(ii+1, n):
            dist = np.linalg.norm(trace_ends[kk, :] - trace_ends[ii, :])
            if dist > maxdist:
                i1 = ii
                i2 = kk
                maxdist = dist

    # Trial vector-a, formed by connecting the two endpoints most distant from each other
    a = trace_ends[[i1, i2], :]
    if a[1, 0] - a[0, 0] < 0:
        a = trace_ends[[i2, i1], :]
    
    a_hat = a[1, :] - a[0, :]
    a_hat = a_hat / np.linalg.norm(a_hat)  # Unit vector
    
    # Projection of end-to-end vector to vector-b_hat
    e = np.zeros(m)
    for jj in range(m):
        e[jj] = np.dot(trace_ends[(jj+1)*2-1, :] - trace_ends[jj*2, :], a_hat)
    
    E = np.sum(e)
    
    # Calculate vector-b with strike discordance corrected
    b = np.array([0., 0.])
    for jj in range(m):
        if np.sign(e[jj]) == np.sign(E):
            b += trace_ends[(jj+1)*2-1, :] - trace_ends[jj*2, :]
        else:
            b -= trace_ends[(jj+1)*2-1, :] - trace_ends[jj*2, :]

    nominal_strike = {
        'a': a,
        'b': b,
        'e': e,
        'E': E
    }
    
    return nominal_strike

# Step 3, Link traces; reverse the strike of discordant trace ----
def linktraces(ftraces, nominal_strike, type, discordant):
    """
    Step 3: Link traces; reverse the strike of discordant trace.

    This function links multiple fault traces, adjusting the strike of any discordant traces
    and calculates the reference axis and origin for coordinate shifting.

    Args:
        ftraces (list of dict): A list of rupture strands, each with 'trace' (ndarray), 'l' (ndarray), and 'strike' (ndarray).
        nominal_strike (dict): A dictionary containing the calculated nominal strike parameters ('a', 'b', 'e', 'E').
        type (dict): A dictionary with 'str' (method for coordinate shifting) and 'po' (coordinate origin for shifting, if 'JB').
        discordant (bool): Flag to reverse strike for discordant traces.

    Returns:
        tuple: A tuple containing:
            - single_trace (dict): A dictionary with keys 'strike', 'l', 's', 'ftrace', 'ref_axis' representing the linked fault traces.
            - reference_axis (ndarray): The reference axis used for coordinate shifting.
            - p_origin (ndarray): The coordinate origin for shifting.
    """

    m = len(ftraces)
    a = nominal_strike['a']
    b = nominal_strike['b'].T
    e = nominal_strike['e']
    E = nominal_strike['E']
    
    if discordant:
        for jj in range(m):
            if e[jj] * E < 0:  # reverse strike, current trace is discordant
                n = ftraces[jj]['trace'].shape[0]
                n1 = n - 1
                ftraces[jj]['trace'] = np.flipud(ftraces[jj]['trace'])
                ftraces[jj]['l'] = np.fliplr(ftraces[jj]['l'])
                ftraces[jj]['strike'] = ftraces[jj]['strike'] - 180
                # ftraces[jj]['p1'] = ftraces[jj]['trace'][0, :]  # This line is commented out in MATLAB

    # Reference axis and origin for calculating coordinate shift (this should be in nominal_strike)
    if type['str'] == 'NGA2':
        reference_axis = np.sign(E) * (a[1, :] - a[0, :])
        if E < 0:
            p_origin = a[1, :]
        else:
            p_origin = a[0, :]
    else:  # Default case from OFR, also used for 'JB'
        if np.dot(a[1, :] - a[0, :], b) >= 0:
            p_origin = a[0, :]
        else:
            p_origin = a[1, :]
        reference_axis = b.T

    reference_axis = reference_axis / np.linalg.norm(reference_axis, 2)

    # Compute Uprime_p1
    Uprime_p1 = np.full(m, np.nan)
    for jj in range(m):
        Uprime_p1[jj] = np.dot(ftraces[jj]['trace'][0, :] - p_origin, reference_axis.T)

    Trace = []
    s = []
    Strike = []
    Len = []
    for jj in range(m):
        ftr = ftraces[jj]
        Trace.append(ftr['trace'])  # Link current trace
        Strike.extend(ftr['strike'] + [np.nan])  # Merge strikes
        Len.extend(ftr['l'] + [np.nan])  # Merge segment lengths
        s.extend(Uprime_p1[jj] + [0] + np.cumsum(ftr['l']).tolist())  # Merge s
        s[-1] = np.nan

    # Remove the last element which is a nan
    Strike = Strike[:-1]
    Len = Len[:-1]
    s = s[:-1]

    single_trace = {
        'strike': Strike,
        'l': Len,
        's': s,
        'ftrace': np.vstack(Trace),
        'ref_axis': reference_axis
    }

    return single_trace, reference_axis, p_origin

def comp_segment_tuw(origin, strike, l, Site):
    """
    Compute (t, u, w) with respect to fault coordinate axes defined by (origin, strike, and l).

    Args:
        origin (ndarray): The origin of the fault trace as [X, Y].
        strike (float): The strike of the fault in degrees.
        l (float): The length of the fault segment.
        Site (dict): A dictionary containing 'StaX' and 'StaY' for the site coordinates.

    Returns:
        dict: A dictionary containing:
            - 't': The distance along the fault in the perpendicular direction.
            - 'u': The distance along the fault in the strike direction.
            - 'wgt': The weight associated with the segment based on the fault and site position.
    """

    # Convert strike to radians
    strikerad = strike / 180 * np.pi

    # Unit vectors for strike (u) and perpendicular (t) directions
    uhat = [np.sin(strikerad), np.cos(strikerad)]
    that = [np.sin(strikerad + np.pi / 2), np.cos(strikerad + np.pi / 2)]

    # Compute t and u
    t = (Site['StaX'] - origin[0]) * that[0] + (Site['StaY'] - origin[1]) * that[1]
    u = (Site['StaX'] - origin[0]) * uhat[0] + (Site['StaY'] - origin[1]) * uhat[1]

    # Compute the weight using the closed-form solution (Equation 1 of the OFR)
    if np.isnan(t):
        wgt = np.nan
    elif abs(t) > 1E-6:  # Rule 1
        wgt = (np.arctan((l - u) / t) - np.arctan(-u / t)) / t
    elif u < 0 or u > l:  # Rule 2
        wgt = 1 / (u - l) - 1 / u
    else:  # Rule 3; T=0
        wgt = np.inf

    # Return the results in a dictionary
    seg_tuw = {
        't': t,
        'u': u,
        'wgt': wgt
    }

    return seg_tuw


def computeGC2(site, single_trace):
    """
    Compute GC2 (T, U, W) for a single trace.

    Args:
        site (dict): Site coordinates with keys 'StaX' and 'StaY'.
        single_trace (dict): Contains the trace details, including 'strike', 'l', 's', and 'ftrace'.

    Returns:
        tuple: A tuple containing:
            - T (numpy.ndarray): The T values for each segment.
            - U (numpy.ndarray): The U values for each segment.
            - W (numpy.ndarray): The weights for each segment.
    """

    strike = single_trace['strike']
    l = single_trace['l']
    s = single_trace['s']
    nseg = len(l)
    p_origin = single_trace['ftrace']
    p_origin = p_origin[:nseg, :]  # nseg x 2 matrix (discards the last [nseg+1] coordinate)

    # Compute site's (t, u, wgt) w.r.t. each of the nseg coordinate systems
    seg_tuw_List = []
    for iseg in range(nseg):
        seg_tuw = comp_segment_tuw(p_origin[iseg, :], strike[iseg], l[iseg], site)
        seg_tuw_List.append(seg_tuw)

    GC2_U = 0.
    GC2_T = 0.
    Wgt = 0.

    for iseg in range(nseg):
        if np.isnan(l[iseg]):
            continue  # Skip bogus segments

        seg_tuw = seg_tuw_List[iseg]
        GC2_U += (seg_tuw['u'] + s[iseg]) * seg_tuw['wgt']
        GC2_T += seg_tuw['t'] * seg_tuw['wgt']
        Wgt += seg_tuw['wgt']

    GC2_U /= Wgt
    GC2_T /= Wgt

    # Apply rule #3 to sites located on the fault
    k_onfault = np.where(Wgt == np.inf)[0]
    GC2_T[k_onfault] = 0
    for kk in k_onfault:
        for ii in range(nseg):
            if np.isinf(seg_tuw_List[ii]['wgt'][kk]):
                GC2_U[kk] = seg_tuw_List[ii]['u'][kk] + s[ii]
                Wgt[kk] = np.nan
                break  # Exit the for loop

    T = GC2_T
    U = GC2_U
    W = Wgt

    return T, U, W


def GC2(ftraces, SiteX, SiteY, trace_type, discordant, gridflag):
    """
    Compute GC2 (T, U, W) for given fault traces and site coordinates.

    Args:
        ftraces (list of dict): A list of N rupture strands. Each element contains:
            - 'trace' (ndarray): A n x 2 array, where n is the number of Cartesian X,Y coordinates defining the surface trace for the strand [km].
            - 'l' (ndarray): A 1 x (n-1) array, defining the length of each segment of the strand [km].
            - 'strike' (ndarray): A 1 x (n-1) array, defining the strike of each segment of the strand [degrees].
        SiteX (ndarray): A 1 x S array, defining the Cartesian X coordinates for which to calculate U and T [km].
        SiteY (ndarray): A 1 x S array, defining the Cartesian Y coordinates for which to calculate U and T [km].
        type (dict): A dictionary with the following fields:
            - 'str' (str): Method used for calculating the coordinate shift ('NGA', 'JB', or other). Defaults to the OFR version.
            - 'po' (ndarray, optional): A 1 x 2 array, specifying the coordinate system origin shift if 'str' is 'JB'.
        discordant (bool): Flag for checking segment discordance (True or False).
        gridflag (bool): Flag for calculating coordinates on a grid (True), or at element-wise coordinates for SiteX and SiteY (False).

    Returns:
        tuple:
            - T (ndarray): GC2 parameter T, either 1 x S or 1 x N, depending on gridflag [km].
            - U (ndarray): GC2 parameter U, either 1 x S or 1 x N, depending on gridflag [km].
            - W (ndarray): GC2 parameter W, either 1 x S or 1 x N, depending on gridflag [unitless].
            - reference_axis (ndarray): The nominal strike direction after correcting for segment discordance, a 1 x 2 unit vector.
            - p_origin (ndarray): The coordinate system origin [km], a 1 x 2 array.
            - nominal_strike (dict): A dictionary with the following fields:
                - 'a' (ndarray): Trial vector-a, formed by connecting the two endpoints most distant from each other, 2 x 2 array [km].
                - 'b' (ndarray): The nominal strike coordinates, 1 x 2 array [km].
                - 'e' (ndarray): The directional dot product for each trace to check for discordance, m x 1 array.
                - 'E' (float): The sum of 'e', a scalar value.
            - Upo (float): The U parameter at the coordinate type.po (before any shift), a scalar value [km].
            - Tpo (float): The T parameter at the coordinate type.po (before any shift), a scalar value [km].
            - gradT (None): This has not yet been implemented.
    """

    # (1) Compute the nominal strike
    nominal_strike = comp_nominal_strike(ftraces)

    # (2) Link the traces and get reference axis and origin
    single_trace, reference_axis, p_origin = linktraces(ftraces, nominal_strike, trace_type, discordant)

    # (3) Compute GC2 for stations
    if gridflag:  # Calculate coordinates on a grid defined by SiteX and SiteY
        T = np.zeros((len(SiteY), len(SiteX)))
        U = np.zeros((len(SiteY), len(SiteX)))
        W = np.zeros((len(SiteY), len(SiteX)))
        for ii in range(len(SiteX)):
            for jj in range(len(SiteY)):
                site = {'StaX': SiteX[ii], 'StaY': SiteY[jj]}
                T[jj, ii], U[jj, ii], W[jj, ii] = computeGC2(site, single_trace)
    else:  # Calculate point-wise for SiteX(1:S) and SiteY(1:S)
        T = np.zeros(len(SiteX))
        U = np.zeros(len(SiteX))
        W = np.zeros(len(SiteX))
        for ii in range(len(SiteX)):
            site = {'StaX': SiteX[ii], 'StaY': SiteY[ii]}
            T[ii], U[ii], W[ii] = computeGC2(site, single_trace)

    # (4) Compute GC2 for the location defined by trace_type['po'] and apply the shift to U
    if trace_type['str'] == 'JB':
        site = {'StaX': trace_type['po'][0], 'StaY': trace_type['po'][1]}
        Tpo, Upo, _ = computeGC2(site, single_trace)
        U = U - Upo
        T = T - Tpo
    else:
        Upo = 0
        Tpo = 0

    # Return the results
    return T, U, W, reference_axis, p_origin, nominal_strike, Upo, Tpo

