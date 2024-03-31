"""
Author: Anuj Mishra

A Bilby source file for performing parameter estimations assuming isolated point lens model.

For an efficient computation of the lensing amplification factor, we adopt following methodology:
1. Precompute the "wave optics" part in a lookup table. 
2. Use Cython to compute the "geometrical-optics" part with the power of C.   
3. Use a dynamic frequency cutoff depenidng on the impact parameter to shift from wave to geometrical optics. This cutoff uses a numerical fit such that the relative errors between the analytic F(f) and the one obtained from geometrical optics limit at any frequency above the cutoff is less than 1%.

For step 1, it assumes a look up table has already been generated before using this source file. The script to generate the lookup table can be found at 'GWMAT/pnt_Ff_lookup_table/pnt_Ff_lookup_table_gen.py'. Once generated, add the path to the lookup table in the object 'lookup_table_file' below.

The function that uses lookuptable to generate F(f), named `pnt_Ff_lookup_table`, can handle any lens mass given it is within the impact parameter range used to build the lookup table. However, a lens mass range of (1e-1, 1e5) is usually sufficient.
"""

import numpy as np
import pickle
from scipy.interpolate import interp1d

from .source import lal_binary_black_hole

# loading Cython version of the point lens class
import sys
sys.path.append('/home/anirban.kopty/pkgs/gwmat/pnt_Ff_lookup_table/src/')  #path to the dir containing cython module
import cythonized_pnt_lens_class as pnt_lens_cy


# lookup_table_file = '/home/anirban.kopty/pkgs/gwmat/pnt_Ff_lookup_table/data/point_lens_Ff_lookup_table__y1e-2_5__w1e-4_1.3e4.pkl'
lookup_table_file = '/home/anirban.kopty/pkgs/gwmat/pnt_Ff_lookup_table/data/from_anuj/point_lens_Ff_lookup_table_Geo_relErr_1p0_Mlz_1e-1_1e5_ys_1e-3_5.pkl'
# lookup_table_file = '/home/anuj.mishra/git_repos/GWMAT/pnt_Ff_lookup_table/data/point_lens_Ff_lookup_table_Geo_relErr_1p0_Mlz_1e-1_1e5_ys_1e-3_5.pkl'
# lookup_table_file = '/home/anuj.mishra/git_repos/git_GWs/pnt_Ff_lookup_table/data/point_lens_Ff_lookup_table_Geo_relErr_1p0_Mlz_1e-1_1e5_ys_1e-3_10.pkl'

## functions
def y_w_grid_data(Ff_grid):
    ys_grid = np.array([Ff_grid[str(i)]['y'] for i in range(len(Ff_grid))])
    ws_grid = Ff_grid[str(np.argmin(ys_grid))]['ws']
    return ys_grid, ws_grid

def y_index(yl, ys_grid):
    return np.argmin(np.abs(ys_grid - yl))

def w_index(w, ws_grid):
    return np.argmin(np.abs(ws_grid - w))


def pnt_Ff_lookup_table(fs, Mlz, yl, Ff_grid, extrapolate=False):
    ys_grid, ws_grid = y_w_grid_data(Ff_grid)

    wfs = np.array([pnt_lens_cy.w_of_f(f, Mlz) for f in fs])
    
    if yl >= 1e-2:
        wc = pnt_lens_cy.wc_geo_re1p0(yl)
    else:
        wc = np.max(wfs)

    wfs_1 = wfs[wfs <= np.min(ws_grid)]
    Ffs_1 =  np.array([pnt_lens_cy.point_Fw(w, y=yl) for w in wfs_1]) 

    wfs_2 = wfs[(wfs > np.min(ws_grid))&(wfs <= np.max(ws_grid))]
    wfs_2_wave = wfs_2[wfs_2 <= wc]
    wfs_2_geo = wfs_2[wfs_2 > wc]

    i_y  = y_index(yl, ys_grid)
    tmp_Ff_dict = Ff_grid[str(i_y)]
    ws = tmp_Ff_dict['ws']
    Ffs = tmp_Ff_dict['Ffs_real'] + 1j*tmp_Ff_dict['Ffs_imag']
    fill_val = ['interpolate', 'extrapolate'][extrapolate]
    i_Ff = interp1d(ws, Ffs, fill_value=fill_val)
    Ffs_2_wave = i_Ff(wfs_2_wave)

    Ffs_2_geo = np.array([pnt_lens_cy.point_Fw_geo(w, yl) for w in wfs_2_geo])

    wfs_3 = wfs[wfs > np.max(ws_grid) ]
    Ffs_3 = np.array([pnt_lens_cy.point_Fw_geo(w, yl) for w in wfs_3])

    Ffs = np.concatenate((Ffs_1, Ffs_2_wave, Ffs_2_geo, Ffs_3))
    assert len(Ffs)==len(fs), 'len(Ffs) = {} does not match len(fs) = {}'.format(len(Ffs), len(fs))
    return Ffs

def _load_lookup_table(lookup_table_file):
    print('## Loading and setting up the lookup table ##')
    with open(lookup_table_file, 'rb') as f:
        Ff_grid = pickle.load(f)
    print(f'## Done - Loaded lookup table - {lookup_table_file}##')
    return Ff_grid

def _load_unless_already_loaded(lookup_table_file):
    if 'Ff_grid' not in globals():
        global Ff_grid
        Ff_grid = _load_lookup_table(lookup_table_file)


### S1. Source Model for point lens microlensing using lookup table ###
def point_lens_MicL_BBH(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                          phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, Log_Mlz, yl, **kwargs):
    """
    This is a Bilby Frequency Domain Source model for performing parameter estimations assuming isolated point lens model.

    It returns microlensed FD plus and cross polarized WFs interpolated at the required frequency_array.

    Parameters
    ----------
    * frequency_array : float 
        Array of frequency values where the WF will be evaluated.        
    * mass_1 : float 
        The mass of the primary component object in the binary (in solar masses).
    * mass_2 : float 
        The mass of the secondary component object in the binary (in solar masses).
    * a_1 : float, optional
        The dimensionless spin magnitude of object 1. 
    * a_2 : float, optional
        The dimensionless spin magnitude of object 2.
    * tilt_1 : ({0.,float}), optional
        Angle between L and the spin magnitude of object 1.
    * tilt_2 : float, optional
        Angle between L and the spin magnitude of object 2.
    * phi_12 : float, optional
        Difference between the azimuthal angles of the spin of the object 1 and 2. 
    * phi_jl : float, optional
        Azimuthal angle of L on its cone about J.     
    * Log_Mlz : float
        Redshifted Mass of the point-lens in log10 scale (in solar masses).
    * yl : float
        The dimensionless impact parameter between the lens and the source.        
    * theta_jn : float, optional
        Angle between the line of sight and the total angular momentum J.   
    * luminosity_distance : ({100.,float}), optional
        Luminosity distance to the binary (in Mpc).
    * theta_jn : float
        Inclination (rad), defined as the angle between the orbital angular momentum J(or, L) and the
        line-of-sight at the reference frequency. Default = 0.              
    * phase : ({0.,float}), optional
        Coalesence phase of the binary (in rad).

    Returns
    -------
    Dictionary:
        * plus: A numpy array.
            Strain values of the plus polarized WF in Frequency Domain..
        * cross: A numpy array.
            Strain values of the cross polarized WF in Frequency Domain.

    """ 
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    _load_unless_already_loaded(lookup_table_file)

    try:
        lal_res = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                  phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)

        Mlz = np.power(10, Log_Mlz)
        if round(Mlz, 3) == 0:
            return dict(plus=lal_res['plus'], cross=lal_res['cross']) 
        else:
            Ff = pnt_Ff_lookup_table(fs=frequency_array, Mlz=Mlz, yl=yl, Ff_grid=Ff_grid)
            lhp = Ff*lal_res['plus']
            lhc = Ff*lal_res['cross']
            return dict(plus=lhp, cross=lhc)
    except Exception as e:
        print(e)
        pass


### S2. Source Model for point lens microlensing using Analytic func + Geo Optics approx. (more accurate but slower; not recommended for PE) ###
def point_lens_MicL_BBH_analytic(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                          phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, Log_Mlz, yl, **kwargs):
    """
    This is similar to above source model "point_lens_MicL_BBH" but uses Analytic function to compute the wave optics part and doesn't
    require a lookup table. This is more accurate but slower, hence not recommended for doing an extensive PE.

    """ 
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    try:
        lal_res = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                  phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)

        Mlz = np.power(10, Log_Mlz)	
        if round(Mlz, 3) == 0:
            return dict(plus=lal_res['plus'], cross=lal_res['cross'])
        else:
            Ff = np.array([pnt_lens_cy.point_Ff_eff(f, ml=Mlz, y=yl) for f in frequency_array])
            lhp = Ff*lal_res['plus']
            lhc = Ff*lal_res['cross']
            return dict(plus=lhp, cross=lhc)

    except Exception as e:
        print(e)
        pass



### S1. Relative Binning version ###
def point_lens_MicL_BBH_relative_binning(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                          phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, Log_Mlz, yl, fiducial, **kwargs):
    """
    See `point_lens_MicL_BBH` doc
    """ 
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    _load_unless_already_loaded(lookup_table_file)

    if fiducial == 0:
        frequency_array = waveform_kwargs["frequency_bin_edges"]

    try:
        lal_res = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                  phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)

        Mlz = np.power(10, Log_Mlz)
        if round(Mlz, 3) == 0:
            return dict(plus=lal_res['plus'], cross=lal_res['cross']) 
        else:
            Ff = pnt_Ff_lookup_table(fs=frequency_array, Mlz=Mlz, yl=yl, Ff_grid=Ff_grid)
            lhp = Ff*lal_res['plus']
            lhc = Ff*lal_res['cross']
            return dict(plus=lhp, cross=lhc)
    except Exception as e:
        print(e)
        pass


### S2. Relative Binning version ###
def point_lens_MicL_BBH_analytic_relative_binning(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                          phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, Log_Mlz, yl, fiducial, **kwargs):
    """
    See `point_lens_MicL_BBH_analytic` doc
    """
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)

    if fiducial == 0:
        frequency_array = waveform_kwargs["frequency_bin_edges"]

    try:
        lal_res = lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
                  phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)

        Mlz = np.power(10, Log_Mlz)	
        if round(Mlz, 3) == 0:
            return dict(plus=lal_res['plus'], cross=lal_res['cross'])
        else:
            Ff = np.array([pnt_lens_cy.point_Ff_eff(f, ml=Mlz, y=yl) for f in frequency_array])
            lhp = Ff*lal_res['plus']
            lhc = Ff*lal_res['cross']
            return dict(plus=lhp, cross=lhc)

    except Exception as e:
        print(e)
        pass
