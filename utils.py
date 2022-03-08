import numpy as np
import healpy as hp
import pandas as pd

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