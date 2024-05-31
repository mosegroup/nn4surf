# This module contains utility functions shared between scripts.

# <<< import external modules <<<
import numpy as np
import torch
import random
import time
import os
# === import external modules ===

def init_profile_gaussian_dimple(
    domain  :   np.ndarray,
    volume  :   float,
    hbase   :   float
    ) -> np.ndarray :
    '''
    This genarates a gaussian profile with a given volume under the curve (+ base)
    '''
    
    L       = domain.max() - domain.min()
    sigma   = L/4
    
    profile = np.zeros( domain.shape )
    
    volume  = volume
    
    for shift in [-L, 0, L]:
        profile += np.exp( -(domain + shift - domain.mean())**2/(2*sigma**2) )
    
    profile = volume * profile/profile.mean() + hbase # renormalize the integral of the profile
    
    return profile

    
def read_dat_profile(
    path    :   str
    ) -> np.ndarray:
    '''
    This returns the profile at path (.dat extension)
    '''
    data = np.loadtxt(path)
    return data[:,2]


def read_npy_profile(
    path    : str,
    ) -> np.ndarray:
    '''
    This returns the profile at path (.npy extension)
    '''
    data = np.load(path).astype(float) # convert to 64bit float in case there is number precision mismatch
    return data[:,1]
