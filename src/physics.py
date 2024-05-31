# <<< importing stuff <<<
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import os
import torch
import random
# === importing stuff ===


# <<< global constants <<<

# sorry for the global use... a future cleanup will solve this

# SiGe interface constants
gamma_Ge    = 6        # Ge [ev/nm2]
gamma_Si    = 8.7      # Si
d_gamma     = 0.27      # denominator in exponential term in wetting energy
at_vol      = 1
M           = 5 # (Tomado de la tesis de Rovaris)

#Elastic constants
strain          = 0.04
young           = 103*6.25 # Ge [eV/nm^3]
poisson         = 0.26  # Ge

CRIT_LAMBDA     = np.pi*gamma_Ge*(1-poisson**2)/young/strain**2
MAX_LAMBDA      = 4/3*crit_lambda

lame_lambda     = young*poisson/(1+poisson)/(1-2*poisson)
lame_mu         = young/2/(1+poisson)

b = 0.01
L = 100

# === global constants ===

def derpar(hprof1,dx):
    '''
    First derivative implementation (finite central difference)
    '''
    narr=hprof1.shape[-1]
    hf=0.0*hprof1
    hf[0,0,0]=(hprof1[0,0,1]-hprof1[0,0,narr-1])/(2*dx)
    hf[0,0,1:narr-1]=(hprof1[0,0,2:narr]-hprof1[0,0,0:narr-2])/(2*dx)
    hf[0,0,narr-1]=(hprof1[0,0,0]-hprof1[0,0,narr-2])/(2*dx)

    return hf


def derpar2(hprof1,dx):
    '''
    Second derivative implementation (finite central difference)
    '''
    narr=hprof1.shape[-1]
    hf2=0.0*hprof1
    hf2[0,0,0]=(hprof1[0,0,1]-2*hprof1[0,0,0]+hprof1[0,0,narr-1])/(dx**2)
    hf2[0,0,1:narr-1]=(hprof1[0,0,2:narr]-2*hprof1[0,0,1:narr-1]+hprof1[0,0,0:narr-2])/(dx**2)
    hf2[0,0,narr-1]=(hprof1[0,0,0]-2*hprof1[0,0,narr-1]+hprof1[0,0,narr-2])/(dx**2)

    return hf2

def gamma(h):
    '''
    Surface energy (as a function of h)
    '''
    return gamma_Ge + (gamma_Si - gamma_Ge)*torch.exp(-h/d_gamma)

def dgamma_dh(h):
    '''
    gamma derivative
    '''
    return -((gamma_Si - gamma_Ge)/d_gamma)*torch.exp(-h/d_gamma)

def kappagamma(h, dx):
    '''
    Curvature contribution to the chemical potential
    '''
    return -gamma(h)*derpar2(h, dx)/(1 + derpar(h, dx)**2)**(3/2)

def mu_wet(h, dx):
    '''
    This is the full wetting contribution
    '''
    return dgamma_dh(h)*(1/torch.sqrt(1 + derpar(h, dx)**2))
