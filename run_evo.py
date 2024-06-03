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
from src.argparser import EvolutionParser
from src.classes import give_NN_model
from src.physics import kappagamma, mu_wet, derpar, derpar2
from src.utils import save_args, np_to_tensor, reload_dat_profile, reload_npy_profile, save_state, initial_rand_profile
# === import n4surf modules ===


def main():
    # The main function; torch is used as backend (parallelization, GPU usage possibility), but in the future numba/compiled stuff may be considered
    
    argparser   = EvolutionParser()
    args        = argparser.parse_args()

    torch.set_grad_enabled(False) # get rid of gradient calculations: we are no longer training
    
    out_path    = f'{args.output_folder}/{args.name}'
    
    # <<< define material constants <<<
    M            = args.M                 # mobility factor (from Fabrizio's thesis)
    path_model   = args.model_path    # path at which the model is saved
    path_profile = args.path_profile  # path to reload profile (set to None if you want a random profile)
    flux_fn      = lambda x: args.flux_value     # This is a constant for now. In the future, we will support custom python callables for flux and analytical terms in evolution
    # === define material constants ===
    
    # <<< time integration variables <<<
    dx          = args.dx           # spatial discretization
    dt          = args.dt           # time integration step
    tot_steps   = args.tot_steps    # total number of steps
    log_freq    = args.log_freq     # logging frequency
    L           = args.L            # domain length
    device      = args.device       # device used for running the simulation
    # === time integration variables ===
    
    # <<< define initial conditions <<<
    num_fourier_components  = args.num_fourier_components   # number of random fourier components
    init_amplitude          = args.init_amplitude           # this is the amplitude of the initial profile
    init_mean               = args.init_mean                # this is the mean for the initial profile
    amplitude_check         = args.amplitude_check          # this is the amplitude check value
    # === define initial conditions  ===

    # <<< profile initialization <<<
    x_numpy   = np.arange(L)
    if path_profile.upper() == 'NONE':
        y   = initial_rand_profile(x_numpy, num_fourier_components, init_amplitude, init_mean)
    elif path_profile.endswith('.npy'):
        y   = reload_npy_profile(path_profile)
    elif path_profile.endswith('.dat'):
        y   = reload_dat_profile(path_profile)
    else:
        raise ValueError(f'Input file {path_profile} has not a valid format.')

    if path_profile.upper() != 'NONE' and args.L != y.shape[-1]:
        raise ValueError(f'The L value {L} is not consistent with the reloaded profile shape {y.shape[-1]}.')
    # === profile intialization ===
    
    # === define initial conditions ===
    
    # <<< convert to torch tensors <<<
    x = np_to_tensor(x_numpy, device)
    y = np_to_tensor(y, device)
    # === convert to torch tensors ===
    
    # <<< load model <<<
    mu_elastic = give_NN_model(path_model, device) # we don't use model_wrapper as we are working in plain torch
    # === load model ===
    
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
