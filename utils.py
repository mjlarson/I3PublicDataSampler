import numpy as np
import healpy as hp
import pandas as pd
import numba

def read_to_pandas(files):
    """
    Helper function to read data release csv files and dump them to pandas
    dataframe objects.

    Args:
        files: A list of files to read. Must have identical formatting.

    Returns:
        A pandas dataframe object concatenating information from all files
    """
    output = []
    for f in files:
        names = open(f,'r').readline().split()[1:]
        output.append(pd.read_csv(f, sep='\s+', names=names, skiprows=1))
    return pd.concat(output)

def bin_to_healpix(nside, ra_deg, dec_deg):
    """
    Convert a list of events to a HEALPix map of counts per bin
    Args:
      nside: A healpix nside
      ra_deg, dec_deg: Equatorial coordinates of the events in degrees (!)
      
    Returns:
      hpx_map: HEALPix map of the counts in in Equatorial coordinates
    """
    indices = hp.ang2pix(nside, ra_deg, dec_deg, lonlat=True)
    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    npix = hp.nside2npix(nside)
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts

    return hpx_map

def ang_dist(src_ra, src_dec, ra, dec):
    """ 
    Compute angular distance between source and location 

    Args:
       src_ra, src_dec: 
           Coordinates of the source in radians
       ra, dec: 
           np.ndarrays giving best-fit positions for each event in radians.

    Returns:
        np.ndarray containing angular distance between each event and the source
    """

    sinDec = np.sin(dec)
    cos_dec = np.sqrt(1. - sinDec**2)
    cosDist = (np.cos(src_ra - ra) * np.cos(src_dec) * cos_dec +
                np.sin(src_dec) * sinDec)

    # handle possible floating precision errors
    cosDist[np.isclose(cosDist, -1.) & (cosDist < -1)] = -1.
    cosDist[np.isclose(cosDist, 1.) & (cosDist > 1)] = 1.
    return np.arccos(cosDist)

@numba.njit(cache=True)
def histogram3d(logemin, logemax,
                sigmamin, sigmamax,
                psimin, psimax,
                llh_values,
                thinning = 5):
    """
    Build a 3d signal PDF from binned IRFs and the weighted expectations,
    including the necessary phase space terms.

    Args:
      logemin, logemax: 
         Numpy arrays containing the lower and upper bin edges in energy
      sigmamin, sigmamax: 
         Numpy arrays containing the lower and upper bin edges in sigma
      psimin, psimax: 
         Numpy arrays containing the lower and upper bin edges in true angular error
      weights:
         The weights associated with each bin divided by the phase space terms
      thinning: 
         A step value used to reduce the number of bins. Applied equally to
           all dimensions as x[::thinning]. The maximum value for each dimension
           is always appended.

    Returns:
        A list of likelihood values for each value
    """
    loge_bins = np.concatenate((np.unique(logemin)[::thinning], 
                                np.array([logemax.max(),])))
    sigma_bins = np.concatenate((np.unique(sigmamin)[::thinning], 
                                 np.array([sigmamax.max(),])))
    psi_bins = np.concatenate((np.unique(psimin)[::thinning], 
                               np.array([psimax.max(),])))
    
    hist = np.zeros((len(loge_bins)-1,
                     len(sigma_bins)-1,
                     len(psi_bins)-1), dtype=np.float64)
    
    emin = np.searchsorted(loge_bins, logemin, 'right')-1
    emax = np.searchsorted(loge_bins, logemax, 'right')-1
    smin = np.searchsorted(sigma_bins, sigmamin, 'right')-1
    smax = np.searchsorted(sigma_bins, sigmamax, 'right')-1
    pmin = np.searchsorted(psi_bins, psimin, 'right')-1
    pmax = np.searchsorted(psi_bins, psimax, 'right')-1
    
    probs = np.exp(llh_values)
    
    for i in range(len(emin)):
        hist[emin[i]:emax[i],
             smin[i]:smax[i],
             pmin[i]:pmax[i]] += probs[i]

    return hist, (loge_bins, sigma_bins, psi_bins)