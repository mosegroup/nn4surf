# <<< import numerical stuff <<<
import numpy as np
import numpy.fft as fft
import torch
import torch.nn as nn
# === import numerical stuff ===

# <<< import utility stuff <<<
import matplotlib.pyplot as plt
import time
import os
import random
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
# === import utility stuff ===

# <<< import mu_NN stuff <<<
from src.classes import *
from src.utils import kappagamma, mu_wet, derpar, derpar2
# === import mu_NN stuff ===



# <<< global constants <<<
FLOAT_TYPE = float # change to modify format (e.g. np.float16 to reduce memory requirements)
# === global constants ===


def save_state(step, arrays, master_path):
    '''
    This saves arrays as a .npy file
    '''
    arrays_tosave = tuple([array.cpu().squeeze().numpy().astype(FLOAT_TYPE) for array in arrays])
    
    np.save(
        f'{master_path}/snapt_{step}',
        np.column_stack(arrays_tosave)
        )
    
    
def reload_npy_profile(path):
    '''
    Returns the profile saved in .npy file (if it is the only element)
    '''
    profile = np.load(path)
    if profile.ndim != 1:
        profile = profile[:,0]
    return np.squeeze(profile).astype(float)


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


def initial_rand_profile(x, num, a_initial, mean_val):
    '''
    This returns a profile with a given initial amplitude (from Luis)
    '''
    L=max(x)
    N=len(x)

    height_tot=0

    for j in range(num):
        ini_wave=np.random.uniform(low = 15, high = 100)
        desfase=np.random.uniform(low = 0, high =10)
        height = -a_initial*np.cos(2*np.pi/ini_wave*x + desfase)
        height_tot = height_tot+height
    
    length=N
    
    gap = int(np.floor(10))
    
    x = np.array([length-gap-1, length-gap, length-gap+1, gap-1, gap, gap+1])
    y = height_tot[x]

    xp   = [-gap-1, -gap, -gap+1, gap-1, gap, gap+1]
    poly = lagrange(xp, y)
    
    sub_data = range(length-gap,length)
    sub      = range(-gap,0)
    height_tot[sub_data] = Polynomial(poly.coef[::-1])(sub)

    sub_data = range(gap)
    sub      = range(gap)
    height_tot[sub_data] = Polynomial(poly.coef[::-1])(sub) # this is necessary to make the profile periodic, as we are not necessarily using lambdas consistent with domain boundaries

    height_tot -= height_tot.mean()
    height_tot = height_tot*a_initial/(max(height_tot)-min(height_tot))
    height_tot = height_tot + mean_val
    
    return height_tot



def main():
    '''
    The main function; we use torch as this allows for faster parallelization for very big profiles if
    '''
    
    torch.set_grad_enabled(False) # get rid of gradient calculations: we are no longer training
    
    out_path    = 'out/growth_1e-4_64bit'
    
    # <<< define material constants <<<
    gamma_Ge    = 6        # Ge [ev/nm2]
    gamma_Si    = 8.7      # Si
    d_gamma     = 0.27     # denominator in exponential term in wetting energy
    M           = 5        # mobility factor (from Fabrizio's thesis)
    path_model      = 'models/model_paper'     # path at which the model is saved
    path_profile    = None             # path to reload profile (set to None if you want a random profile)
    flux_fn     = lambda x: 1e-4
    # === define material constants ===
    
    # <<< time integration variables <<<
    dx          = 1             # spatial discretization
    dt          = 1e-3          # time integration step
    tot_steps   = 50_000_000    # total number of steps
    log_freq    = 5_000         # logging frequency
    L           = 100           # domain length
    device      = 'cpu'         # device used for running the simulation
    # === time integration variables ===
    
    # <<< define initial conditions <<<
    num_fourier_components  = 15    # number of random fourier components
    init_amplitude          = 1e-3  # this is the amplitude of the initial profile
    init_mean               = 0.0   # this is the mean for the initial profile
    amplitude_check         = 1e-2  # this is the amplitude check value

    # -- profile initialization --
    x_numpy   = np.arange(L)
    if path_profile is None:
        y   = initial_rand_profile(x_numpy, num_fourier_components, init_amplitude, init_mean)
    elif path_profile.endswith('.npy'):
        y   = reload_npy_profile(path_profile)
    elif path_profile.endswith('.dat'):
        y   = reload_dat_profile(path_profile)
    else:
        raise ValueError(f'Input file {path_profile} has not a valid format.')
    # -- profile intialization --
    # === define initial conditions ===
    
    # <<< convert to torch tensors <<<
    x = np_to_tensor(x_numpy, device)
    y = np_to_tensor(y, device)
    # === convert to torch tensors ===
    
    # <<< load model <<<
    mu_elastic = give_NN_model(path_model, device) # we don't use model_wrapper as we are working in plain torch
    mu_elastic.double()
    # === load model ===
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    else:
        raise FileExistsError('Output folder already exists! Remove it or rename ouptut')
    
    os.system(f'cp run_evo.py {out_path}/run_evo.py')
    
    for step in range(tot_steps): # this is the dynamics loop
        
        start_loop  = time.time()
        
        mu_gamma    = kappagamma(y, dx)
        mu_wetting  = mu_wet(y, dx)
        mu_elas     = mu_elastic(y-y.mean())
        
        mu_tot      = mu_gamma + mu_wetting + mu_elas
        
        flux = flux_fn(step)
        
        # check the amplitude of the profile
        if y.max()-y.min() < amplitude_check:
            # in this case we need to resize everything
            y = initial_rand_profile(x_numpy, num_fourier_components, 2*amplitude_check, mean_val=y.mean().item())
            y = np_to_tensor(y, device)

        y += (M * derpar( derpar(mu_tot,dx)/torch.sqrt(1+derpar(y,dx)**2), dx ) + flux)*dt
        
        end_loop    = time.time()
        loop_time   = end_loop - start_loop #  this is the time required for the individual loop
        
        
        if step % log_freq == 0:
            save_state(step, (y,mu_elas), out_path)
            print(f'Step {step}/{tot_steps} \t flux = {flux} \t material = {y.mean()} \t loop_time: {loop_time*1000:.2f}ms') 

if __name__ == '__main__':
    main()
