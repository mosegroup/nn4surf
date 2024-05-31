# This module contains utility functions shared between scripts.

# <<< import external modules <<<
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import torch
import random
import time
import os
# === import external modules ===

# <<< import nn4surf modules <<<
from src.physics import MAX_LAMBDA # definition of constants/quantities
# === import nn4surf modules ===


# <<< specify float type <<<
# these are different in case in the future we want to use different precision in calculation and saving (e.g. to save memory)
FLOAT_TYPE_CALC = float
FLOAT_TYPE_SAVE  = float
# === specift float type ===

# <<< profile initialization utils <<<

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


def initial_rand_profile(
    x           : np.ndarray,
    num         : int,
    a_initial   : float,
    mean_val    : float
    ) -> np.ndarray:
    '''
    This returns a profile with a given initial amplitude (from Luis)
    '''
    L   = x.max() - x.min()
    N   = len(x)

    height_tot  = 0

    for j in range(num):
        ini_wave    = np.random.uniform(low = 15, high = 100)
        desfase     = np.random.uniform(low = 0, high =10)
        height      = -a_initial*np.cos(2*np.pi/ini_wave*x + desfase)
        height_tot  = height_tot+height
    
    gap     = int(np.floor(10))
    
    x       = np.array([N-gap-1, N-gap, N-gap+1, N-1, gap, gap+1])
    y       = height_tot[x]

    xp      = [-gap-1, -gap, -gap+1, gap-1, gap, gap+1]
    poly    = lagrange(xp, y)
    
    sub_data    = range(N-gap,N)
    sub         = range(-gap,0)
    height_tot[sub_data] = Polynomial(poly.coef[::-1])(sub)

    sub_data    = range(gap)
    sub         = range(gap)
    height_tot[sub_data] = Polynomial(poly.coef[::-1])(sub) # this is necessary to make the profile periodic, as we are not necessarily using lambdas consistent with domain boundaries

    height_tot  -= height_tot.mean()
    height_tot   = height_tot*a_initial/(max(height_tot)-min(height_tot))
    height_tot   = height_tot + mean_val
    
    return height_tot

# === profile initialization utils  ===


# <<< log utils <<<

def save_args(path, args):
    # simply save args at path
    with open(path, 'w+') as out_file:
        for arg in dir(args):
            if arg[0] != '_': # we don't need to save __name_, etc.
                out_file.write( f'{arg} \t : \t {vars(args)[arg]}\n' )


def save_state(step, arrays, master_path):
    '''
    This saves arrays as a .npy file
    '''
    arrays_tosave = tuple([array.cpu().squeeze().numpy().astype(FLOAT_TYPE_SAVE) for array in arrays])
    
    np.save(
        f'{master_path}/snapt_{step}',
        np.column_stack(arrays_tosave)
        )

# === log utils ===
    
    
# <<< type conversion <<<

def np_to_tensor(array, device):
    '''
    This converts array to device and put to tensor if needed
    '''
    if not isinstance(array, torch.Tensor):
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array).double()
        else:
            raise ValueError('The passed array is not a numpy array.')
        
    while array.ndim < 3: # add required dimension to be run on model
        array = array.unsqueeze(0)
        
    array = array.to(device) # put to the required device
    
    return array

# === type conversion ==


# <<< reading utils <<<

def reload_dat_profile(path, formatting='short'):
    '''
    Returns the profile saved in dat file
    '''
    data = np.loadtxt(path)
    if formatting == 'long':
        return data[:-1:20, 2] # this serves as a direct interface with FEM output calculations
    elif formatting == 'short':
        return data[:,2]
    elif formatting == 'NN':
        return data[:,3]
    elif isinstance(formatting, str):
        raise ValueError(f'Value {formatting} for opening format encountered in loading .dat file is not valid.')
    else:
        raise ValueError(f'Offending formatting value in loading .dat file (type {type(formatting)}).')


def reload_npy_profile(path):
    '''
    Returns the profile saved in .npy file (if it is the only element)
    '''
    profile = np.load(path)
    if profile.ndim != 1:
        profile = profile[:,0]
    return np.squeeze(profile).astype(FLOAT_TYPE_CALC) # casting to right dimension in case of different precision

# === reading utils ===
