#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:55:56 2023

@author: egemo
"""

from torch.cuda import is_available
from sys import stdout

def get_device(p=False):
    """
    Simple helper that returns "cuda" if a cuda-available device is present, 
    else "cpu"

    Parameters
    ----------
    p : boolean, optional
        If True, the function prints an output on screen. The default is False.

    Returns
    -------
    device : str
        Variable used in pytorch to initialize the tensors.

    """ 
    device = "cuda" if is_available() else "cpu"
    if p:
        print("using {} device".format(device))
    return device

def progress(count, total, status=''):
    # source: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    stdout.flush()

class Dict(dict):
    """
    This class inherits all the functionalities of a python class,
    and adds the possibility to use the dot notation to access to the 
    instance content.
    It also includes a few useful functions that return or modify existing 
    elements.
    
    For example:
    conventional dict           Dict (this class)
    a = dict()                  a = Dict()
    a['b'] = 1                  a['b'] = 1
    c = a['b']                  c = a.b    
    """
    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    def first(self):
        """
        Retuns the instance first element, in order of declaration.
        """
        return self[next(iter(self))]
    def update_existing(self,other_dict):
        """
        This function takes the elements in the other_dict dictionary, and 
        repladces the self dictionary items with the other_dict items ONLY
        if matching keys are present

        Parameters
        ----------
        other_dict : dict or Dict
            dictionary from which the items may be taken to replace the self 
            items.

        Returns
        -------
        None.

        """
        for k,i in other_dict.items():
            if k in self:
                self[k] = i
        # self.update((k, other_dict[k]) for k in set(Dict2).intersection(self))
    def update_excluding(self,other_dict,excludedkeyslist):
        """
        This function updates the self dictionary with the elements in 
        other_dict, only if the key of other_dict is not present in the list
        excludedkeyslist

        Parameters
        ----------
        other_dict : dict or Dict
            dictionary from which the items may be taken to replace the self 
            items.
        excludedkeyslist : list (of keys)
            list.

        Returns
        -------
        None.

        """
        for k,i in other_dict.items():
            if not (k in excludedkeyslist): 
                self[k] = i
        # self.update({x: other_dict[x] for x in other_dict if x not in excludedkeyslist})
        
        
###############################################################################
### torch functions modifications
from torch import (rand,
                   zeros,
                   ones)
def uniform_dist(*args,min=-ones(1),max=ones(1),**kwargs):
    """
    Overrides the torch.rand function and adds the min-max values functionality
    CAREFUL - no assertion is done on the min and max values
    
    Parameters
    ----------
    *args : any
        arguments of the torch.rand function.
    min : numerical or tensor, optional
        minimum value (left boundary) of the uniform distribution. The default is -1.
    max : numerical or tensor, optional
        max value (right boundary) of the uniform distribution. The default is 1.
    **kwargs : any
        keyword arguments for the torch.rand function.

    Returns
    -------
    tensor
        uniform distribution comprised between the min and max values.

    """
    return min+rand(*args,**kwargs)*(max-min)