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

    def __init__(self, table_path, mode='mu', every=20):

        super(TabulatedSeries, self).__init__()
        
        self.mode           = mode
        self.table_path     = table_path # <--- here there is a .txt file containing the path to all individual examples
        
        self.every = every

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
        
        profile, mu_eps, x = data[:-1:self.every,1], data[:-1:self.every,2], data[:-1:self.every,0]
            
        profile, mu_eps, x = self.numpyfy( [profile, mu_eps, x] )
        
        if torch.rand(1).item() <= 0.5:
            x           = x.flip(-1)
            profile     = profile.flip(-1)
            mu_eps      = mu_eps.flip(-1)
            
        profile -= profile.mean(dim=-1, keepdim=True)

        return profile, mu_eps, x
    
    
    
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
        
