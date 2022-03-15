import os, sys, glob
import logging
import numpy as np
import pandas as pd
from scipy.special import logsumexp

import utils

class BackgroundGenerator:
    def __init__(self, data_files, floor=None):
        """
        Class to generate signal events from IceCube's data release files

        Args:
          data_files: A list of files giving events corresponding to one continuous 
              period of datataking (eg, IC86-II+).
          floor: If given, use this number as the minimum for the background histogram
              used to build the pdf. Otherwise, set the minimum counts per bin to 
              1/len(data).
        """
        self.logger = logging.Logger("BackgroundGenerator")
        
        # Read the data files
        self.exp_data = utils.read_to_pandas(data_files)
        
        # Generate bins in sin(dec), energy, ... sigma?
        sindec = np.sin(np.radians(self.exp_data['Dec[deg]'].values))
        self.sindec_bins = np.unique(sindec)
        while len(self.sindec_bins) > 120:
            self.sindec_bins = self.sindec_bins[::2]
        if not self.sindec_bins[-1] == sindec.max():
            self.sindec_bins = np.concatenate([self.sindec_bins,
                                               [sindec.max()*1.01,]])

        # And in energy
        self.loge_bins = np.unique(self.exp_data['log10(E/GeV)'])
        while len(self.loge_bins) > 100:
            self.loge_bins = self.loge_bins[::2]
        if not self.loge_bins[-1] == self.exp_data['log10(E/GeV)'].max():
            self.loge_bins = np.concatenate([self.loge_bins,
                                            [self.exp_data['log10(E/GeV)'].max()*1.01,]])
                
        # Build the histogram we'll use for pdf evaluation
        self.pdf_histogram, _, _ = np.histogram2d(self.exp_data['log10(E/GeV)'],
                                                  sindec,
                                                  bins=(self.loge_bins,
                                                        self.sindec_bins),
                                                  density=False
                                                  )
        if floor is None:
            floor = 1.0/len(self.exp_data)
        else: floor = 0
        self.pdf_histogram[self.pdf_histogram==0] = floor
        self.pdf_histogram /= self.pdf_histogram.sum()
        
        # Scale to make the histogram into a pdf
        phase = np.diff(self.loge_bins)[:,None] * np.diff(self.sindec_bins)[None,:]
        self.pdf_histogram /= phase
        
        #x = (self.loge_bins[:-1] + self.loge_bins[1:])/2.
        #y = (self.sindec_bins[:-1] + self.sindec_bins[1:])/2.
        #self._pdf = RectBivariateSpline(x, y, self.pdf_histogram,
        #                                kx=1, ky=1, s=0)
        return
    
    def pdf(self, loge, sindec):
        """
        Evaluate the interpolated pdf to get a likelihood.
        
        Args:
           loge: np.ndarray-like array of reconstructed energies
           sindec:  np.ndarray-like array of reconstructed sin(declination)
           
        Returns:
           Likelihood values for each event
        """
        #return self._pdf(loge, sindec)
        i = np.searchsorted(self.loge_bins, loge, side='right')-1
        j = np.searchsorted(self.sindec_bins, sindec, side='right')-1
        
        i = np.clip(i, 0, len(self.loge_bins)-2)
        j = np.clip(j, 0, len(self.sindec_bins)-2)
        
        return self.pdf_histogram[i,j]
    
    def logpdf(self, loge, sindec):
        """
        Evaluate the interpolated np.log(pdf) to get a likelihood.
        
        Args:
           loge: np.ndarray-like array of reconstructed energies
           sindec:  np.ndarray-like array of reconstructed sin(declination)
           
        Returns:
           Log-likelihood values for each event
        """
        return np.log(self.pdf(loge, sindec))
    
    def scramble(self):
        """
        Generate a background trial by copying the dataset and randomizing
        the right ascensions values
        
        Returns:
            Pandas dataframe containing simulated background events of the form
               pd.DataFrame({"ra":ra, "dec":dec, "sigma":sigma, "logE":logE}) 
        """
        new_ra = np.random.uniform(0, 2*np.pi, len(self.exp_data))
        
        return pd.DataFrame({"ra":new_ra, 
                             "dec":np.radians(self.exp_data['Dec[deg]']),
                             "sigma":np.radians(self.exp_data['AngErr[deg]']),
                             "logE":self.exp_data['log10(E/GeV)']})