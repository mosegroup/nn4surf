# <<< import stuff <<<
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import glob
import random
import torch.nn.utils.parametrize as parametrize
# --- import stuff ---

'''
This file contains classes and other utilities for training and running the model
'''


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
    
    model_for_params = convmodel(
        kernel_size     = 21, # these are hardcoded for now... in future versions they will be no longer
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
    model = torch.jit.trace( model.forward, torch.randn(1,1,100).to(device) ) # traced objects are faster
    
    return model
