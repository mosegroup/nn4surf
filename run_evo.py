# <<< import numerical modules <<<
import numpy as np
import numpy.fft as fft
import torch
import torch.nn as nn
# === import numerical modules ===

# <<< import utility modules <<<
import matplotlib.pyplot as plt
import time
import os
import random
# === import utility modules ===

# <<< import nn4surf modules <<<
from src.classes import give_NN_model
from src.physics import kappagamma, mu_wet, derpar, derpar2
from src.utils import save_args, np_to_tensor, reload_dat_profile, reload_npy_profile, save_state
# === import n4surf modules ===


def main():
    # The main function; torch is used as backend (parallelization, GPU usage possibility), but in the future numba/compiled stuff may be considered
    
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
    # === define initial conditions  ===

    # <<< profile initialization <<<
    x_numpy   = np.arange(L)
    if path_profile is None:
        y   = initial_rand_profile(x_numpy, num_fourier_components, init_amplitude, init_mean)
    elif path_profile.endswith('.npy'):
        y   = reload_npy_profile(path_profile)
    elif path_profile.endswith('.dat'):
        y   = reload_dat_profile(path_profile)
    else:
        raise ValueError(f'Input file {path_profile} has not a valid format.')
    # === profile intialization ===
    
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
