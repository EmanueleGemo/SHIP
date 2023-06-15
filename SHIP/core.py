#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:54:30 2023

@author: egemo
"""

from .utils import get_device
from torch import arange

class core:
    """
    Easy way to broadcast var change would be to modify these placeholders.
    At the moment, all those are shadowed
    """
    batch_size = 1 
    nts = 1
    dt = 1
    device = get_device()
    
    @classmethod
    def timeline(self,lim = None):
        """
        Helper that returns the temporal sequence followed during the inference
        (to be changed in following versions)

        Parameters
        ----------
        lim : int, optional
            Limits the temporal sequence up to the lim timestep, is stated.
            The default is None.

        Returns
        -------
        torch.tensor()
            Tensor containing the sum of the timesteps up to the last 
            (or lim) timestep.

        """
        if lim and lim<self.nts:
            return arange(lim)*self.dt
        else:
            return arange(self.nts)*self.dt
        
    def sethidden(self,attr,val):
        setattr('_'+self.__class__.__name__+attr,val)
    def gethidden(self,attr):
        return getattr('_'+self.__class__.__name__+attr)
