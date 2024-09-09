# <<< import stuff <<<
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import glob
import random
import time
# === import stuff ===


class TabulatedSeries(torch.utils.data.Dataset):
    '''
    This is a proper implementation of a dataset in pytorch
    '''

    def __init__(self, table_path, mode='mu', replicas=0, every=20):

        super(TabulatedSeries, self).__init__()
        
        self.mode           = mode
        self.table_path     = table_path # <--- here there is a .txt file containing the path to all individual examples
        
        self.every = every
        self.replicas = replicas

        with open(self.table_path,'r') as table_file:
            table  = table_file.readlines()
        
        self.table = [line[:-1] for line in table]
        del table
        
        self.length = len(self.table)
        
    def __len__(self): 
    
        return self.length
    
    
    def readdata(self, path):
        if self.mode == 'mu':
            data = np.loadtxt(path, skiprows=1, usecols=(0,1,2))
        elif self.mode == 'strain':
            data = np.loadtxt(path, skiprows=1, usecols=(0,1,3,5)) # load x,y,exx,eyy (exy determined by normal stress conditions)
        elif self.mode == 'strain3':
            data = np.loadtxt(path. skiprows=1, usecols((0,1,3,4,5)))
        else:
            raise NotImplementedError(f'{self.mode} mode is not implemented yet')
        return data
    
    
    def numpyfy(self, ll):
        out = []
        for l in ll:
            out.append( torch.from_numpy(l).float().unsqueeze(0) )
            
        return out
    
        
    def __getitem__(self, idx):
        
        line = self.table[idx]
        data = self.readdata(line)

        if self.mode == 'mu':
            
            profile, mu_eps, x = data[:-1:self.every,1], data[:-1:self.every,2], data[:-1:self.every,0]
            
            if self.replicas > 1:
                profile = np.tile(profile, self.replicas)
                mu_eps  = np.tile(mu_eps, self.replicas)
                dx = x[1]-x[0]
                x = np.linspace(0,mu_eps.shape[-1]*dx,mu_eps.shape[-1])

            profile, mu_eps, x = self.numpyfy( [profile, mu_eps, x] )
            
            if torch.rand(1).item() <= 0.5:
                x           = x.flip(-1)
                profile     = profile.flip(-1)
                mu_eps      = mu_eps.flip(-1)
                
            profile -= profile.mean(dim=-1, keepdim=True)

            return profile, mu_eps, x

        elif self.mode == 'strain':

            profile, epsxx, epsyy, x = data[:-1:self.every,1], data[:-1:self.every,2], data[:-1:self.every,3], data[:-1:self.every,0]

            if self.replicas > 1:
                raise NotImplementedError(f'Replicas are not implemented yet in strain mode.')

            profile, epsxx, epsyy, x = self.numpyfy( [profile, epsxx, epsyy, x] )

            eps = torch.cat((epsxx, epsyy), dim=0)

            if torch.rand(1).item() <= 0.5:
                x           = x.flip(-1)
                profile     = profile.flip(-1)
                eps         = eps.flip(-1)

            profile -= profile.mean(dim=-1, keepdim=True)
            return profile, eps, x

        elif self.mode == 'strain3':

            profile, epsxx, epsxy, epsyy, x = data[:-1:self.every,1], data[:-1:self.every,2], data[:-1:self.every,3], data[:-1:self.every,4], data[:-1:self.every,0]

            if self.replicas > 1:
                raise NotImplementedError(f'Replicas are not implemented yet in strain mode.')

            profile, epsxx, epsxy, epsyy, x = self.numpyfy( [profile, epsxx, epsxy, epsyy, x] )

            eps = torch.cat((epsxx, epsxy, epsyy), dim=0)
            
            if torch.rand(1).item() <= 0.5:
                x           = x.flip(-1)
                profile     = profile.flip(-1)
                eps         = eps.flip(-1)

            profile -= profile.mean(dim=-1, keepdim=True)
            return profile, eps, x

        else:
            raise ValueError(f'It seems that training set loading mode is {self.mode}. Something nasty may be going on...')
    

    
class TabulatedSeries_strain(torch.utils.data.Dataset):
    '''
    This is a proper implementation of a dataset in pytorch
    '''

    def __init__(self, table_path, mode='mu'):

        super().__init__()
        
        self.mode           = mode
        self.table_path     = table_path # <--- here there is a .txt file containing the path to all individual examples

        with open(self.table_path,'r') as table_file:
            table  = table_file.readlines()
        
        self.table = [line[:-1] for line in table]
        del table
        
        self.length = len(self.table)
        
    def __len__(self): 
    
        return self.length
    
    
    def readdata(self, path):
        if self.mode == 'mu':
            data = np.loadtxt(path, skiprows=1, usecols=(0,1,3,4,5))
        else:
            raise NotImplementedError(f'{self.mode} mode is not implemented yet')
        return data
    
    
    def numpyfy(self, ll):
        out = []
        for l in ll:
            if len(l.shape) == 1:
                out.append( torch.from_numpy(l).float().unsqueeze(0) )
            else:
                out.append( torch.from_numpy(l).float() )
            
        return out
    
        
    def __getitem__(self, idx):
        
        line = self.table[idx]
        data = self.readdata(line)
        
        (profile, strain, x) = (data[:-1:20,1], data[:-1:20,2:], data[:-1:20,0])
        
        strain = strain.T
        
        profile, strain, x = self.numpyfy( [profile, strain, x] )
        
        
        
        if torch.rand(1).item() <= 0.5:
            x           = x.flip(-1)
            profile     = profile.flip(-1)
            strain      = strain.flip(-1)
            
        profile -= profile.mean(dim=-1, keepdim=True)

        return profile, strain, x
        
