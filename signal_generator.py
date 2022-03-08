import os, sys, glob
import logging
import numpy as np
import pandas as pd
import healpy as hp
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

import numba
import utils

@numba.njit(cache=True, fastmath=True)
def logaddexp(a, b):
    if b == -np.inf: return a
    if a == -np.inf: return b
    if b-a > 3: return b
    if a-b > 3: return a
    m = max(a,b)
    return np.log(np.exp(a-m)+np.exp(b-m))
    
@numba.njit(cache=True, fastmath=True, parallel=True)
def histogram3d(values, loge_bins, sigma_bins, psf_bins, weights):
    hist = np.ones((len(loge_bins), len(sigma_bins), len(psf_bins))) * -np.inf
    shape = hist.shape
    
    for m in range(values.shape[1]):
        a = np.searchsorted(loge_bins, values[0,m], side='right')-1
        b = np.searchsorted(sigma_bins, values[1,m], side='right')-1
        c = np.searchsorted(psf_bins, values[2,m], side='right')-1
        hist[a,b,c] = logaddexp(hist[a,b,c], weights[m])

    #dloge = np.log(np.diff(loge_bins))
    #dsigm = np.log(2*np.pi * -np.diff(np.cos(sigma_bins)))
    #dpsf = np.log(2*np.pi * -np.diff(np.cos(psf_bins)))
            
    for i in range(1, shape[0]):
        for j in numba.prange(shape[1]):
            for k in numba.prange(shape[2]):
                hist[i,j,k] = logaddexp(hist[i,j,k], hist[i-1,j,k])
    for i in numba.prange(shape[0]):
        for j in range(1, shape[1]):
            for k in numba.prange(shape[2]):
                hist[i,j,k] = logaddexp(hist[i,j,k], hist[i,j-1,k])
        for j in numba.prange(shape[1]):
            for k in range(1, shape[2]):
                hist[i,j,k] = logaddexp(hist[i,j,k], hist[i,j,k-1])
                    
    return np.exp(hist)



@numba.njit(cache=True)
def get_llh(loge, logemin, logemax,
            sigma, sigmamin, sigmamax,
            psi, psimin, psimax,
            llh_values):
    output = np.full_like(loge, -np.inf)

    for j in range(len(loge)):
        for i in range(len(logemin)):
            if ((logemin[i] <= loge[j]) & (loge[j] < logemax[i])
                & (sigmamin[i] <= sigma[j]) & (sigma[j] < sigmamax[i])
                & (psimin[i] <= psi[j]) & (psi[j] < psimax[i])):
                output[j] = np.logaddexp(output[j], llh_values[i])
    return output

class SignalGenerator:
    def __init__(self, aeff_files, smearing_files, uptime_files,
                 angular_resolution_scale=1.0):
        """
        Class to generate signal events from IceCube's data release files

        Args:
          aeff_files: A list of files giving effective areas for one continuous
              period of datataking (eg, IC86-II+). Generally will only contain a
              single file.
          smearing_files: A list of files giving instrument response tables for 
              one continuous period of datataking (eg, IC86-II+). Generally will 
              only contain a single file.
          uptime_files: A list of files giving detector uptimes for corresponding 
              to one continuous period of datataking (eg, IC86-II+).
          angular_resolution_scale: A scale factor used to rescale IceCube's point
              spread function. Useful when testing eg impact of improving resolutions
              or generating events from KM3NeT/P-ONE (set to 0.5) instead of IceCube
              (set to 1.0). 
        """
        self.logger = logging.Logger("SignalGenerator")
        
        # Total livetime for IceCube
        uptime = utils.read_to_pandas(uptime_files)
        self.total_uptime = (uptime['MJD_stop[days]'] - uptime['MJD_start[days]']).sum()
        del uptime

        # Calculate the exposure
        aeff = utils.read_to_pandas(aeff_files)
        aeff['A_Eff[cm^2]'] *= self.total_uptime * 24*3600.
        aeff['E_nu/GeV_min'] = 10**aeff['log10(E_nu/GeV)_min']
        aeff['E_nu/GeV_max'] = 10**aeff['log10(E_nu/GeV)_max']
        aeff = aeff.drop(labels=['log10(E_nu/GeV)_min','log10(E_nu/GeV)_max'], axis=1)
        exposure = aeff.rename(mapper={'A_Eff[cm^2]':"Exposure[cm^2 s]"}, axis=1)
        self.exposure = exposure.groupby(['Dec_nu_min[deg]','Dec_nu_max[deg]'])
        del aeff

        # And finally get the transfer matrix
        smearing = utils.read_to_pandas(smearing_files)
        if angular_resolution_scale != 1:
            smearing['PSF_min[deg]'] *= angular_resolution_scale
            smearing['PSF_max[deg]'] *= angular_resolution_scale
        smearing['E_nu/GeV_min'] = 10**smearing['log10(E_nu/GeV)_min']
        smearing['E_nu/GeV_max'] = 10**smearing['log10(E_nu/GeV)_max']
        smearing = smearing.drop(labels=['log10(E_nu/GeV)_min','log10(E_nu/GeV)_max'], axis=1)
        
        # Add a column corresponding to the bin phase space
        logphase = np.log(smearing['log10(E/GeV)_max'] - smearing['log10(E/GeV)_min'])
        logphase = np.logaddexp(logphase, np.log(2*np.pi*(np.cos(np.radians(smearing['AngErr_min[deg]'])) 
                                                 - np.cos(np.radians(smearing['AngErr_max[deg]'])))))
        logphase = np.logaddexp(logphase, np.log(2*np.pi*(np.cos(np.radians(smearing['PSF_min[deg]']))
                                                 - np.cos(np.radians(smearing['PSF_max[deg]'])))))
        smearing['log_phase_space'] = logphase
        
        self.transfer = smearing.groupby(['Dec_nu_min[deg]', 'Dec_nu_max[deg]'])
        del smearing

        return

    def create_ra_dec(self, psi, src_ra, src_dec):
        """
        Take an array of angular reconstruction errors and the
        true RA/dec of the sources and generate reconstructed
        directions.

        Args:
          psi: A list of (true, not estimated) angular errors associated 
               with events in radians
          src_ra, src_dec: Equatorial coordinates of the origin of each
               signal neutrino in radians

        Returns:
          reco_ra, reco_dec: Reconstructed right ascension, declination
               for each signal event in radians
        """
        psi = np.atleast_1d(psi)
        src_ra = np.atleast_1d(src_ra)
        src_dec = np.atleast_1d(src_dec)    # Use the true RA and dec here to get a unit vector
        true_unit = np.array([np.cos(src_ra)*np.cos(src_dec),
                              np.sin(src_ra)*np.cos(src_dec),
                              np.sin(src_dec)])    # Add a rotation in right ascension
        R = Rotation.from_euler('Z', psi)
        reco_unit = R.apply(true_unit.T)    # Choose a rotation angle around the truth
        theta = np.random.uniform(0, 2*np.pi, size=len(psi))
        S = np.sin(theta/2)
        quaternion = np.array([true_unit[0]*S,
                               true_unit[1]*S,
                               true_unit[2]*S,
                               np.cos(theta/2)]).T
        R = Rotation.from_quat(quaternion)
        reco_unit = R.apply(reco_unit).T    # Convert back to RA/Dec
        reco_ra = np.arctan2(reco_unit[1],
                             reco_unit[0])
        reco_dec = np.arcsin(reco_unit[2])
        return reco_ra, reco_dec
    
    def get_expectations(self, 
                         declination_deg, right_ascension_deg,
                         integral_flux_function):
        """
        Generate expected signal counts in reco space from a point source following 
        the given (integral) flux model.

        Args:
          declination_deg: 
             The declination of your source in degrees
          right_ascension_deg: 
             The RA of your source in degrees
          integral_flux_function: 
             A callable function giving the functional form of the integral
             with respect to energy of your selected flux model. Requires the
             function signature
                f(energy_min, energy_max) --> ndarray[float]
             with output fluxes in units of 1/cm2/s
 
        Returns:
           Transfer matrix with `N_expected` column giving the number of events
           expected from this flux.
        """        
        # Get the total expected number of events
        # Loop until we get the right one, then stop the loop
        for dec_bin, exposure in self.exposure:
            if (dec_bin[0] <= declination_deg) & (declination_deg < dec_bin[1]):
                break
                        
        # Next step: divide the total up to per reco bin
        # Loop until we get the right one, then stop the loop
        for dec_bin, transfer in self.transfer:
            if (dec_bin[0] <= declination_deg) & (declination_deg < dec_bin[1]):
                break
                                
        # Correct transfer matrix bin. How many events should we generate from each bin?
        # We want to get the average exposure over this energy bin. Nominally, that's
        #   avg_exposure = \int_{emin}^{emax}(exposure dE) / \int_{emin}^{emax} dE
        # Using trapezoidal rule,
        #   avg_exposure = ((emax - emin) * 0.5 * (exposure(emax) - exposure(emin))
        #                    / (emax - emin))
        #   avg_exposure = 0.5 * (exposure(emax) - exposure(emin))
        # ie, we just take the averages evaluated at the transfer energy bin edges.
        transfer_exposure = 0.5 * (np.interp(transfer['E_nu/GeV_max'], 
                                             exposure['E_nu/GeV_max'],
                                             exposure['Exposure[cm^2 s]'])
                                   + np.interp(transfer['E_nu/GeV_min'], 
                                               exposure['E_nu/GeV_max'],
                                               exposure['Exposure[cm^2 s]']))
        
        N_expected = (transfer_exposure * transfer['Fractional_Counts']
                      * integral_flux_function(transfer['E_nu/GeV_min'], 
                                               transfer['E_nu/GeV_max']))
       
        # Drop zeros
        transfer = transfer[N_expected>0]
        N_expected = N_expected[N_expected>0]

        # Print and return
        N_tot = N_expected.sum()
        self.logger.debug(f"Expecting {N_tot} signal events from the flux given.")
        return transfer, N_expected
    
    def _sample_uniform(self, low, high, n):
        """
        Helper function to sample [n0, n1, n2, ...] values uniformly between
        [low0, low1, low2, ...] and [high0, high1, high2, ...].
        
        Args:
          low: np.ndarray containing lower bounds per bin
          high: np.ndarray containing upper bounds per bin
          n: np.ndarray containing number to sample from each bin
 
        Returns:
           np.ndarray containing samples uniformly drawn from input
        """
        return np.random.uniform(low.repeat(n), high.repeat(n))

    def generate_pointsource(self, 
                             declination_deg, right_ascension_deg,
                             integral_flux_function):
        """
        Generate signal events from a point source following the given
        (integral) flux model. 
        
        Assumptions:
            - Reconstruction values can be sampled uniformly across each transfer 
              matrix bin (ie, can choose logE uniformly from [logE_min^{bin}, logE_max^{bin}])

        Args:
          declination_deg: 
             The declination of your source in degrees
          right_ascension_deg: 
             The RA of your source in degrees
          integral_flux_function: 
             A callable function giving the functional form of the integral
             with respect to energy of your selected flux model. Requires the
             function signature
                f(energy_min, energy_max) --> ndarray[float]
             with output fluxes in units of 1/cm2/s
 
        Returns:
           Pandas dataframe containing simulated signal events of the form
               pd.DataFrame({"ra":ra, "dec":dec, "sigma":sigma, "logE":logE}) 
        """
        transfer, N_expected = self.get_expectations(declination_deg, 
                                                     right_ascension_deg,
                                                     integral_flux_function)
        N_observed = np.random.poisson(N_expected)
        self.logger.debug(f"Generating {N_observed.sum()} signal events from the flux given.")
                        
        # Start creating an event
        # Assign energy and sigma
        logE = self._sample_uniform(transfer['log10(E/GeV)_min'],
                                    transfer['log10(E/GeV)_max'],
                                    N_observed)
        sigma = self._sample_uniform(transfer['AngErr_min[deg]'], 
                                     transfer['AngErr_max[deg]'],
                                     N_observed)
        sigma = np.radians(sigma)

        # And the harder part: generating reco RA/dec values...
        psi = self._sample_uniform(transfer['PSF_min[deg]'], 
                                   transfer['PSF_max[deg]'],
                                   N_observed)
        ra, dec = self.create_ra_dec(np.radians(psi),
                                     np.radians(right_ascension_deg),
                                     np.radians(declination_deg))
        return pd.DataFrame({"ra":ra, "dec":dec, "sigma":sigma, "logE":logE})

    def pdf_pointsource(self, 
                        events,
                        declination_deg, right_ascension_deg,
                        integral_flux_function,
                        new=False):
        """
        Calculate a signal likelihood for a given set of events using energy, 
        estimated angular uncertainty, and angular distance.

        Args:
          events: 
             A pandas dataframe containing events you'd like to evaluate. 
             Assumed to minimally have the following columns 
                pd.DataFrame({"ra":ra, "dec":dec, "sigma":sigma, "logE":logE})
          declination_deg: 
             The declination of your source in degrees
          right_ascension_deg: 
             The RA of your source in degrees
          integral_flux_function: 
             A callable function giving the functional form of the integral
             with respect to energy of your selected flux model. Requires the
             function signature
                f(energy_min, energy_max) --> ndarray[float]
             with output fluxes in units of 1/cm2/s

        Returns:
            A list of likelihood values for each value
        """
        # Calculate the angular distance from each event to the source
        dist = utils.ang_dist(np.radians(right_ascension_deg),
                              np.radians(declination_deg),
                              events['ra'], events['dec'])
        dist = np.degrees(dist)

        # Our full likelihood space is defined by the signal expectations
        # assuming our given flux model
        transfer, N_expected = self.get_expectations(declination_deg, 
                                                     right_ascension_deg,
                                                     integral_flux_function)
        
        # Calculate the phase space. We'll need these
        # Query is picky. Temp rename columns
        transfer_pdf = np.log(N_expected.values/N_expected.sum())

        # Gather values from the transfer matrix
        loge = np.concatenate([transfer['log10(E/GeV)_min'], transfer['log10(E/GeV)_max']])
        loge_bins = np.unique(loge)
        sigma = np.concatenate([transfer['AngErr_min[deg]'], transfer['AngErr_max[deg]']])
        sigma_bins = np.unique(sigma)
        psf = np.concatenate([transfer['PSF_min[deg]'], transfer['PSF_max[deg]']])
        psf_bins = np.unique(psf)

        # Histogram them
        values = np.array([loge, sigma, psf])
        weights = np.concatenate([transfer_pdf, -transfer_pdf])
        hist = histogram3d(values, loge_bins, sigma_bins, psf_bins, weights)
        
        phase = (np.diff(loge_bins)[:,None,None]
                 * -2*np.pi * np.diff(np.cos(np.radians(sigma_bins)))[None,:,None]
                 * -2*np.pi * np.diff(np.cos(np.radians(psf_bins)))[None,None,:])
        print(np.diff(loge_bins))
        print(-2*np.pi * np.diff(np.cos(np.radians(sigma_bins))))
        print(-2*np.pi * np.diff(np.cos(np.radians(psf_bins))))
        hist[:-1,:-1,:-1] *= phase

        # And the lookups.
        loge_i = np.searchsorted(loge_bins, events.logE.values, side='right')-1
        sigma_i = np.searchsorted(sigma_bins, np.degrees(events.sigma.values), side='right')-1
        psf_i = np.searchsorted(psf_bins, dist.values, side='right')-1

        return hist[loge_i, sigma_i, psf_i]
    
    def generate_diffuse(self, 
                         integral_flux_function):
        """
        Generate signal events from a unresoloved/diffuse following the given
        (integral) flux model. 

        This function will generate signal events for each declination band
        available in the effective area  by calling the following for each
        declination bin [decmin, decmax]
            temp = lambda emin, emax: (4*pi * (sin(decmax)-sin(decmin))
                                       * integral_flux_function(emin, emax))
            dec_events = generate_pointsource(dec, 0, temp)
        Once events are generated, they will be rotated to randomly selected
        values in right ascension.

        Args:
          integral_flux_function: 
             A callable function giving the functional form of the integral
             with respect to energy of your selected flux model. Requires the
             function signature
                f(energy_min, energy_max) --> ndarray[float]
             with output fluxes in units of 1/cm2/s/sr
 
        Returns:
           Pandas dataframe containing simulated signal events of the form
               pd.DataFrame({"ra":ra, "dec":dec, "sigma":sigma, "logE":logE}) 
        """
                
        # Start looping over declination bands to generate events
        signal_events = []
        for dec, rows in self.exposure:
            # Create the necessary integral flux shim function
            solid_angle = 2*np.pi*(np.sin(np.radians(dec[1]))-np.sin(np.radians(dec[0])));
            temp_integral_flux_function = lambda emin, emax: (solid_angle 
                                                              * integral_flux_function(emin, emax))
            # Generate the events with true RA = 0
            events = self.generate_pointsource(dec[0], 0, temp_integral_flux_function)
            
            # And rotate events to real RA values
            true_ras = np.random.uniform(0, 2*np.pi, size = len(events))
            events['ra'] = np.mod(events['ra'].values + np.radians(true_ras), 2*np.pi)
            
            # The true declinations of each event are at the lower bin edge at
            # the moment. Smooth the distribution in dec by moving the true 
            # declinations around within the bin
            sindec = np.sin(np.radians(dec))
            dec_shifts = (np.arcsin(np.random.uniform(sindec[0], sindec[1], len(events))) 
                          - np.radians(dec[0]))
            events['dec'] += dec_shifts
            signal_events.append(events)

        return pd.concat(signal_events)
    
    def generate_from_healpix(self, 
                              healpix_map,
                              integral_flux_function):
        """
        Generate signal events from a point source following the given
        (integral) flux model by pulling normalization values from a 
        provided healpix map. Note that additional normalizations can
        be included inside the integral_flux_function to scale the simulated
        flux relative to the map if desired.
        
        This function will generate signal events for each declination band
        by summing the total flux from the healpix band, F_{tot}. Events will
        be generated by calling
            temp = lambda emin, emax: integral_flux_function(F_{tot}, emin, emax)
            dec_events = generate_pointsource(dec, 0, temp)
        Once events are generated, they will be rotated from 0 to RA values selected
        using np.random.choice from the list of values in the healpix dec band.

        Args:
          healpix_map: 
             A healpy map giving flux normalizations for each point on the sky.
             Units here don't matter much: the values will be passed to the 
             integral_flux_function to be interpreted there.
          integral_flux_function: 
             A callable function giving the functional form of the integral
             with respect to energy of your selected flux model. Requires the
             function signature
                f(healpy_value, energy_min, energy_max) --> ndarray[float]
             with output fluxes in units of 1/cm2/s
        Returns:
           Pandas dataframe containing simulated signal events of the form
               pd.DataFrame({"ra":ra, "dec":dec, "sigma":sigma, "logE":logE}) 
        """
        npix = len(healpix_map)
        nside = hp.npix2nside(npix)
        ra_deg, dec_deg = hp.pix2ang(nside, np.arange(npix), lonlat=True)
        healpix_dataframe = pd.DataFrame({'ra_deg':ra_deg, 'dec_deg':dec_deg, 'flux':healpix_map})
        del ra_deg, dec_deg
        
        
        # Start looping over declination bands to generate events
        signal_events = []
        for dec, rows in healpix_dataframe.groupby('dec_deg'):
            total = rows['flux'].values.sum()
                    
            # Create the necessary integral flux shim function
            temp_integral_flux_function = lambda emin, emax: integral_flux_function(total, emin, emax)
            
            # Generate the events with true RA = 0
            events = self.generate_pointsource(dec, 0, temp_integral_flux_function)
            
            # And rotate events to real RA values
            true_ras = np.random.choice(rows['ra_deg'].values,
                                        size = len(events),
                                        p = rows['flux'].values/rows['flux'].values.sum())
            
            events['ra'] = np.mod(events['ra'].values + np.radians(true_ras), 2*np.pi)
            signal_events.append(events)

        return pd.concat(signal_events)