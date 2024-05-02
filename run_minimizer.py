# <<< import stuff <<<
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import os
import torch
import random
from src.classes import * # sorry for wild import, will probably be cleaned up in a future release
from src.utils import *
import time
# === import stuff ===


def init_profile(
    domain  :   np.ndarray,
    volume  :   float,
    hbase   :   float
    ) -> np.ndarray :
    '''
    This genarates a gaussian profile with a given volume under the curve (+ base)
    '''
    
    sigma = max_lambda/2
    L   = domain.max() - domain.min()
    
    profile = np.zeros( domain.shape )
    
    volume  = volume
    
    for shift in [-L, 0, L]:
        profile += np.exp( -(domain + shift - domain.mean())**2/(2*sigma**2) )
    
    profile = volume * profile/profile.mean() + hbase # renormalize the integral of the profile
    
    return profile
    
    
def read_profile(
    path    :   str
    ) -> np.ndarray:
    '''
    This returns the profile at path
    '''
    data = np.loadtxt(path)
    return data[:,2]


class Minimizer:
    '''
    This class implements the minimization procedure of tge F[h] free energy functional
    '''
    def __init__(
        self,
        model_path  :   str, # the path for the NN model to calculate the elastic energy
        device      :   str = None
        ) -> None :
        
        self.set_device(device)
        self.reload_model(model_path)
        
        
    def set_device(
        self,
        device  :   str
        ) -> None :
        '''
        Set the device on which to run the minimization
        '''
        if device == 'cuda':
            self.device = 'cuda'
        elif device == 'cpu' or device is None:
            self.device = 'cpu'
        else:
            raise ValueError(f'device {device} is not a valid key. Only "cuda" and "cpu" ara available.')
        
        
    def reload_model(
        self,
        model_path  :   str
        ) -> None :
        '''
        This function reloads model at path
        '''
        model_for_params = convmodel(
                kernel_size     = 21,
                depth           = 5,
                channels        = 20,
                activation      = nn.Tanh()
                )

        model = convmodel_no_parametrization(
                kernel_size     = 21,
                depth           = 5,
                channels        = 20,
                activation      = nn.Tanh()
                )
        
        assert isinstance(model_path, str)
        
        if not model_path.endswith('.pt'):
            model_path += '.pt'
        
        model_for_params.load_state_dict( torch.load(f'{model_path}') )

        model.to(self.device)
        model_for_params.to(self.device)

        model.net = model_for_params.net
        model.set_symmetry()
        model = torch.jit.trace( model.forward, torch.randn(1,1,100).to(self.device) )
        
        self.model = model
        del model, model_for_params
        
        
    def minimize(
        self,
        profile,              # the seed to use for minimization
        alpha       :   float   = 1e-2, # alpha parameter for minimization
        steps_max   :   int     = 30_000,  # maximum number of iterations
        max_tol     :   float   = 1e-4, # tollerance on the chemical potential
        log         :   int     = 100,
        eigen       :   float   = 0.04 # new eigenstrain
        ) -> np.ndarray:
        '''
        This method minimizes the given profile
        '''
        with torch.no_grad():
            if not isinstance(profile, torch.Tensor):
                assert isinstance(profile, np.ndarray) # else, we will raise an error
                profile = torch.from_numpy( profile ).float()
                
            while profile.ndim < 3:
                profile = profile.unsqueeze(0)
            
            dx = 1
            
            for step in range(steps_max):
                # this is the actual minimization loop
                dF_dh = kappagamma(profile, dx) + mu_wet(profile, dx) + (eigen/0.04)**2*self.model(profile-profile.mean())
                dF_dh = dF_dh - dF_dh.mean() # remove mean value (i.e. out-project the non-conservative component of the "force")
                
                mean_residual = torch.abs( dF_dh ).mean()
                max_residual = torch.abs( dF_dh ).max()
                
                if max_residual <= max_tol:
                    return profile, dF_dh
                else:
                    profile -= alpha*dF_dh
                    
                if step % log == 0:
                    print('='*20)
                    print(f'Minimization step {step}')
                    print(f'Maximum residual {max_residual}')
                    print(f'Mean absolute residual {mean_residual}')
            
            print('Maximum number of iterations reached in minimization process. Returning')
            return profile, dF_dh
            
            
            


def minimize_eigen(name, eigen):
    '''
    Run minimization, exploits the eigenstrain rescaling trick to have a different effective Ge concentration.
    '''
    
    # define domain and initial profile
    x = np.arange(100)
    prof = init_profile(x, 0.05, 1.6)
    prof_0 = prof.copy()
    
    model_path = 'models/model_paper.pt'
    
    # instantiate minimization process
    minimizer = Minimizer( model_path )
    
    prof, mu_end = minimizer.minimize(prof, alpha=5e-3, max_tol=1e-4, eigen=eigen, steps_max=100_000)
    
    np.savetxt(
        f'{name}_eig{eigen}.dat',
        np.column_stack( (x, prof_0.squeeze(), prof.squeeze(), mu_end.squeeze()) ),
        header = 'x\th_0\th'
        )



def minimize_shift(name, shift):
    # define domain and initial profile
    x = np.arange(100)
    #prof = read_profile( 'seed_0_model_paper_dt0.001_1_5Mstep/evo_1/snapt_0.dat' )
    prof = init_profile(x, 0.05, shift)
    prof_0 = prof.copy()
    
    model_path = 'models/model_paper.pt'
    
    # instantiate minimization process
    minimizer = Minimizer( model_path )
    
    prof, mu_end = minimizer.minimize(prof, alpha=5e-3, max_tol=1e-4, eigen=0.04, steps_max=100_000)
    
    np.savetxt(
        f'{name}_shift{shift}.dat',
        np.column_stack( (x, prof_0.squeeze(), prof.squeeze(), mu_end.squeeze()) ),
        header = 'x\th_0\th'
        )




def main():
    name = 'min_small'
    for shift in [2.6,2.7,2.8,2.9,3.0]:
        minimize_shift(name, shift)


if __name__ == '__main__':
    main()
