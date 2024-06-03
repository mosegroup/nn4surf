# <<< import stuff <<<
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
import os
import glob
import random
import torch.nn.utils.parametrize as parametrize
# --- import stuff ---


class FlipSymmetric(nn.Module):
    '''
    This class implements flip-symmetric kernels
    '''
    def __init__(self):
        super().__init__()

    def forward(self, kernel):
        return 0.5*( kernel + kernel.flip(-1) )

class convmodel(nn.Module):
    '''
    The CNN class
    '''
    def __init__(self, kernel_size, depth, channels, activation=nn.Tanh()):
        '''
        Constructor method
        '''
        super().__init__()
        
        self.kernel_size    = kernel_size
        self.depth          = depth
        self.channels       = channels
        self.activation     = activation
        
        netlist = nn.ModuleList()
        
        for kk in range(self.depth):
            if kk == 0:
                in_channels     = 1
            else:
                in_channels     = self.channels
                
            if kk == self.depth-1:
                out_channels    = 1
                activation      = nn.Identity()
            else:
                out_channels    = self.channels
                activation      = self.activation
                
            netlist.append(
                nn.Conv1d(
                    in_channels     = in_channels,
                    out_channels    = out_channels,
                    kernel_size     = self.kernel_size,
                    stride          = 1,
                    padding         = (self.kernel_size-1)//2,
                    padding_mode    = 'circular'
                    )
                )
            netlist.append( activation )
            
        self.net = nn.Sequential(*netlist)
        self.netlist = netlist

        for module in self.netlist:
            if hasattr(module, 'weight'):
                parametrize.register_parametrization( module, 'weight', FlipSymmetric() )
                

    def forward(self, x):
        '''
        The forward method
        '''
        return self.net(x)
    
    

class convmodel_no_parametrization(nn.Module):
    '''
    The CNN class. This a variant without parametrization for symmetric kernels
    '''
    def __init__(self, kernel_size, depth, channels, activation=nn.Tanh()):
        '''
        The constructor
        '''

        super().__init__()
        
        self.kernel_size    = kernel_size
        self.depth          = depth
        self.channels       = channels
        self.activation     = activation
        
        netlist = nn.ModuleList()
        
        for kk in range(self.depth):
            if kk == 0:
                in_channels     = 1
            else:
                in_channels     = self.channels
                
            if kk == self.depth-1:
                out_channels    = 1
                activation      = nn.Identity()
            else:
                out_channels    = self.channels
                activation      = self.activation
                
            netlist.append(
                nn.Conv1d(
                    in_channels     = in_channels,
                    out_channels    = out_channels,
                    kernel_size     = self.kernel_size,
                    stride          = 1,
                    padding         = (self.kernel_size-1)//2,
                    padding_mode    = 'circular'
                    )
                )
            netlist.append( activation )
            
        self.net = nn.Sequential(*netlist)
        self.netlist = netlist

    def set_symmetry(self):
        for module in self.net:
            if hasattr(module, 'weight'):
                module.weight.data = 0.5*( module.weight.data + module.weight.data.flip(-1) )
                
    def forward(self, x):
        return self.net(x)



class model_wrapper:
    '''
    This is a model wrapper utility, abstracting all torch operations and making it more similar to a numpy function
    '''
    
    def __init__(self, path, device='cpu'):
        '''
        The constructor
        '''
        self.device     = device
        self.model      = give_NN_model(path, device=self.device) # here we store the model
        
        
    def __call__(self, x):
        '''
        This is a simple utility intefrace which transforms the input to appropriate shape etc. and returns plain np array
        '''
        with torch.no_grad(): # disable gradient tracking, as we should be in evaluation mode
            
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float().to(self.device)
            elif isinstance(x, list):
                x = torch.tensor(x, device=self.device)
            elif not isinstance(x, torch.Tensor):
                raise ValueError(f"Variable of type f{type(x)} is not permitted in model.")
            
            while len(x.shape) < 3:
                x = x.unsqueeze(0) # now data have the correct shape
            
            predicted_mu = self.model( x - x.mean(dim=-1, keepdim=True) ).squeeze()
            
            return predicted_mu.cpu().numpy()
    
    
    def set_device(self, device):
        '''
        This is a simple setter method
        '''
        self.device = device
        self.model.to(self.device)
        
        
        
def give_NN_model(path, device='cpu'):
    '''
    Returns the NN models with parameters saved in path
    '''
    
    if not path.endswith('.pt'):
        path = path + '.pt'
    
    # hardcoded for now...
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
    
    model_for_params.load_state_dict( torch.load(f'{path}') )
    
    model.net = model_for_params.net
    model.set_symmetry()
    model.to(device)
    model.double()

    model = torch.jit.trace( model.forward, torch.randn(1,1,100, dtype=torch.double).to(device) ) # traced objects are faster

    return model



class Minimizer:
    '''
    This class implements the minimization procedure of tge F[h] free energy functional
    '''
    def __init__(
        self,
        model_path      :   str, # the path for the NN model to calculate the elastic energy
        analytic_terms,
        device          :   str = None
        ) -> None :
        
        self.set_device(device)
        self.reload_model(model_path)
        self.analytic_terms = analytic_terms
        
        
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
        self.model.double() # set model on float64 precision
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
                profile = torch.from_numpy( profile )
                
            while profile.ndim < 3:
                profile = profile.unsqueeze(0)
            
            dx = 1
            
            for step in range(steps_max):
                # this is the actual minimization loop
                dF_dh = self.analytic_terms(profile, dx) + (eigen/0.04)**2*self.model(profile-profile.mean())
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




class sharedUNet(nn.Module):
    # this class implement a special U Net architecture, in which weights are shared at different resolutions

    def __init__(self, kernel_size, depth, channels, path, interpolation_mode='adaptive', activation=nn.Tanh()):
        # the constructor method

        super().__init__()

        self.kernel_size    = kernel_size
        self.depth          = depth
        self.channels       = channels
        self.path           = path
        self.activation     = activation

        self.model1 = give_NN_model(path)
        self.model2 = give_NN_model(path)
        self.model3 = give_NN_model(path)

        self.merger = nn.Conv1d(
                in_channels     = 3,
                out_channels    = 1,
                kernel_size     = self.kernel_size,
                stride          = 1,
                padding         = (self.kernel_size-1)//2,
                padding_mode    = 'circular'
                )

        self.interpolation_mode = interpolation_mode

        if self.interpolation_mode == 'adaptive':

            self.downscaler =  nn.Conv1d(
                    in_channels     = 1,
                    out_channels    = 1,
                    kernel_size     = 4,
                    stride          = 2,
                    padding         = 1,
                    padding_mode    = 'circular'
                    )

            self.upscaler = nn.ConvTranspose1d(
                    in_channels     = 1,
                    out_channels    = 1,
                    kernel_size     = 4,
                    stride          = 2,
                    padding         = 1,
                    )

        elif self.interpolation_mode != 'simple':
            raise ValueError(f'The interpolation mode {self.interpolation_mode} is not valid.')


    def forward(self, x):
        # first, we replicate the input at different resolutions
        x1 = x

        if self.interpolation_mode == 'simple':
            x2 = torchfunc.interpolate( x, size=x.shape[-1]//2)
            x3 = torchfunc.interpolate( x, size=x.shape[-1]//4)
        elif self.interpolation_mode == 'adaptive':
            # perform "adaptive" rescaling
            x2 = self.downscaler(x1)
            x3 = self.downscaler(x2)
        else:
            raise ValueError(f'The interpolation mode {self.interpolation_mode} is not valid.')

        # predict mu at different resolutions
        mu1 = self.model1(x1)
        mu2 = 0.5*self.model2(x2)   # need 1/2 rescaling of prediction
        mu3 = 0.25*self.model3(x3)  # nees 1/4 rescaling of prediction

        # resize the predictions
        if self.interpolation_mode == 'simple':
            mu2 = torchfunc.interpolate( mu2, size=x.shape[-1])
            mu3 = torchfunc.interpolate( mu3, size=x.shape[-1])
        elif self.interpolation_mode == 'adaptive':
            # perform "adaptive" rescaling
            mu2 = self.upscaler(mu2)
            mu3 = self.upscaler(self.upscaler(mu3))
        else:
            raise ValueError(f'The interpolation model {self.interpolation_mode} is not valid.')

        # concatenate predictions
        mus = torch.cat( (mu1,mu2,mu3), dim=1 )

        return self.merger(mus) # perform "weighted average"

