# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:45:24 2023

@author: Em
"""

from torch import (#where,
                   zeros_like,
                   tensor,
                   exp,
                   ones,
                   zeros,
                   is_tensor,
                   #stack,
                   #arange,
                   #abs as t_abs,
                   #autograd,
                   no_grad,
                   logical_and)


##############################################################################
# neuron activator functions

@classmethod
def standard_activator(self,arg): # returns full precision tensor (more flexible)
    out = zeros_like(arg)
    out[arg >= 0] = 1
    return out

@classmethod     
def quick_activator(self,arg): # simplified version - returns bool tensor
    return arg >= 0

##############################################################################
# commmon functions, might be used as scaffold for some models

def LIF_time_dep(self):  
    if is_tensor(self.tau_beta):
        self.dt_tau = self.dt/self.tau_beta
    else:
        self.dt_tau = tensor(self.dt/self.tau_beta)
    self.beta = exp(-self.dt_tau)
    
def LS_time_dep(self):
    if is_tensor(self.tau_alpha):
        self.alpha = (-self.dt/self.tau_alpha).exp()        
    else:
        self.alpha = tensor(-self.dt/self.tau_alpha).exp() 
        
def get_alphas(ta,dt):
    if is_tensor(ta): # data type checking
        A = (-dt/ta).exp();
    else:
        A = tensor(-dt/ta).exp();
    B = ta*(1-A)
    return A,B
    

##############################################################################
# refractory neuron dynamic class

def refractory(parentclass):
    """
    This dynamic class assumes that the neuron model uses the variable
    "integrate" to track the non-integrating time.
    It adds a "refr_time" parameter, upon which the neuron determines how much
    time needs to pass after spiking, before starting to integrate once more.
    CAREFUL - not properly working with PyTorch training - need to debug.
              use refractory_variabletimestep instead

    Parameters
    ----------
    parentclass : group

    Returns
    -------
    childclass : refractory_group

    """
    tag = "refractory_"+parentclass.__name__
    kwargs = {"dynamic_class":True,
              "__init__":refr__init__,
              "time_dep":refr_time_dep,
              "set_initial_state":refr_set_initial_state,
              "advance_timestep":refr_advance_timestep}
    childclass = type(tag,(parentclass,),kwargs)
    return childclass


def refr__init__(self,*args,**kwargs):
    super(self.__class__, self).__init__(*args,**kwargs)
    super(self.__class__, self).arg2InitDict({'refr_table':[]})
    

# @classmethod
def refr_time_dep(self):
    super(self.__class__, self).time_dep()
    # self.refractory_time_dep()
    
    r = self.refr_time
    if not is_tensor(r):
        r =  tensor(r)
    if r.dim() == 0:
        r = ones(self.batch_size,self.N)*r
    elif r.dim() == 1:
        if r.shape == self.N:
            r = r.unsqueeze(0).expand([self.batch_size,self.N])
        elif r.shape == self.batch_size:
            r = r.unsqueeze(1).expand([self.batch_size,self.N])
        else:
            raise Exception('please check refr_time tensor'' shape in %s class'%self.tag) # LOG - algorithm should never enter here, 
    self.refr_steps = (r.div(self.dt)).round().int()
    self.refr_max_steps = self.refr_steps.max().item()
    self.refr_table = ones([self.refr_max_steps,self.batch_size,self.N],dtype = bool)
    self.refr_new = self.refr_table.clone()
    for ii in range(self.batch_size):
        for jj in range(self.N):
            self.refr_new[:self.refr_steps[ii,jj],ii,jj] = False
 
def refr_set_initial_state(self,*args,**kwargs):
    super(self.__class__, self).set_initial_state(*args,**kwargs)
    self.refr_t = 0
    
# @classmethod
def refr_advance_timestep(self,local_input):    
    # update table here:    
        
    spikes = super(self.__class__, self).advance_timestep(local_input)

    # erase previous information
    self.refr_table[self.refr_t,:] = True
    # advance counter
    self.refr_t = (self.refr_t+1)%self.refr_max_steps
    # use spikes to determine new table state
    if spikes.any():
        self.refr_table[:,spikes.detach().bool()] = self.refr_new[:,spikes.detach().bool()].roll(self.refr_t,0)
    
    # retrieve next integrate var from table
    self.integrate = self.refr_table[self.refr_t,:,:]
    
    return spikes


def refractory_variabletimestep(parentclass): 
    """
    TO DO - to be merged with the other refractory superclass
    so, here we use a single-column table to check the numeric value of the 
    out-time, instead of a boolean table.

    Parameters
    ----------
    parentclass : group

    Returns
    -------
    childclass : refractory_group

    """ 
    tag = "refractory_variabletimestep_"+parentclass.__name__
    kwargs = {"dynamic_class":True,
              "__init__":refr_variabletimestep__init__,
              "time_dep":refr_variabletimestep_time_dep,
              "set_initial_state":refr_variabletimestep_set_initial_state,
              "advance_timestep":refr_variabletimestep_advance_timestep}
    childclass = type(tag,(parentclass,),kwargs)
    return childclass


def refr_variabletimestep__init__(self,*args,**kwargs):
    super(self.__class__, self).__init__(*args,**kwargs)
    super(self.__class__, self).arg2InitDict({'refr_table':[]})
    

# @classmethod
def refr_variabletimestep_time_dep(self):
    super(self.__class__, self).time_dep()
    # self.refractory_time_dep()
    
    r = self.refr_time
    if not is_tensor(r):
        r =  tensor(r)
    if r.dim() == 0:
        r = ones(self.batch_size,self.N)*r
    elif r.dim() == 1:
        if r.numel() == self.N:
            r = r.unsqueeze(0).expand([self.batch_size,self.N])
        elif r.numel() == self.batch_size:
            r = r.unsqueeze(1).expand([self.batch_size,self.N])
        else:
            raise Exception('please check refr_time tensor'' shape in %s class'%self.tag) # LOG - algorithm should never enter here, 
    self.on_spike = r

# @classmethod   
def refr_variabletimestep_set_initial_state(self,*args,**kwargs):
    super(self.__class__, self).set_initial_state(*args,**kwargs)
    self.refractive_time = zeros(self.batch_size,self.N)
    
    
# @classmethod
def refr_variabletimestep_advance_timestep(self,local_input):    
    # update table here:    
        
    spikes = super(self.__class__, self).advance_timestep(local_input)
    
    self.refractive_time[spikes.detach().bool()] = self.on_spike[spikes.detach().bool()] 
    self.integrate = self.refractive_time<=0
    self.refractive_time[~self.integrate] = self.refractive_time[~self.integrate]-self.dt
    
    return spikes

##############################################################################
# latency (output delay) neuron/synapse utils

def delayed(parentclass):
    """
    this superclass adds a "delay_time" parameter, which shifts the output
    along the time axix (acting as a buffer, whose memory is the list 
    variable delayed_output)

    Parameters
    ----------
    parentclass : group model

    Returns
    -------
    childclass : delayed_group model

    """
    # 
    
    tag = "delayed_"+parentclass.__name__
    kwargs = {"dynamic_class":True,
              "time_dep":latency_time_dep,
              "set_initial_state":latency_set_initial_state,
              "advance_timestep":latency_advance_timestep}
    childclass = type(tag,(parentclass,),kwargs)
    return childclass

# @classmethod
def latency_time_dep(self):
    super(self.__class__, self).time_dep()
    self.delay_steps = max([round(self.delay_time/self.dt),1])
    self.delayed_output = [zeros([self.batch_size,self.Nt],dtype = self.dtype) for _ in range(self.delay_steps)]
    
# @classmethod   
def latency_set_initial_state(self,*args,**kwargs): 
    super(self.__class__, self).set_initial_state(*args,**kwargs)
    self.delayed_output = [zeros([self.batch_size,self.Nt],dtype = self.dtype) for _ in range(self.delay_steps)]
    
# @classmethod
def latency_advance_timestep(self,local_input):
    self.delayed_output.append( super(self.__class__, self).advance_timestep(local_input) )
    return self.delayed_output.pop(0)

# def delayed_variabletimestep(parentclass):
#     # as above - for variable time stepping (TO DO - to be merged with the above)
#     tag = "delayed_variabletimestep_"+parentclass.__name__
#     kwargs = {"dynamic_class":True,
#               "time_dep":variabletimestep_latency_time_dep,
#               "set_initial_state":variabletimestep_latency_set_initial_state,
#               "advance_timestep":variabletimestep_latency_advance_timestep}
#     childclass = type(tag,(parentclass,),kwargs)
#     return childclass

# # @classmethod
# def variabletimestep_latency_time_dep(self):
#     super(self.__class__, self).time_dep()
#     self.delay_steps = max([round(self.delay_time/self.dt),1])
#     self.delayed_output = [zeros([self.batch_size,self.Nt],dtype = self.dtype) for _ in range(self.delay_steps)]
    
# # @classmethod   
# def variabletimestep_latency_set_initial_state(self,*args,**kwargs): 
#     super(self.__class__, self).set_initial_state(*args,**kwargs)
#     self.delayed_output = [zeros([self.batch_size,self.Nt],dtype = self.dtype) for _ in range(self.delay_steps)]
    
# # @classmethod
# def variabletimestep_latency_advance_timestep(self,local_input):
#     self.delayed_output.append( super(self.__class__, self).advance_timestep(local_input) )
#     return self.delayed_output.pop(0)

##############################################################################
# winner-take-all utility for neurons

def WinnerTakeAll(parentclass):
    """
    This superclass applies the winner-take-all functionality to a neurongroup
    Each time a neuron of the neurongroup spikes, the membrane potential of 
    all the neurons in the group is reset (in the following timestep).

    Parameters
    ----------
    parentclass : neurongroup model

    Returns
    -------
    childclass : WTA_neurongroup model

    """
    tag = "WTA_"+parentclass.__name__
    kwargs = {"dynamic_class":True,
              "advance_timestep":WTA_advance_timestep}
    childclass = type(tag,(parentclass,),kwargs)
    return childclass

# @classmethod
def WTA_advance_timestep(self,local_input):
    output = super(self.__class__, self).advance_timestep(local_input)    
    with no_grad(): # TODO: experimental - might need to review this one
        toreset = output.any(dim=1)
        if toreset.any():
            self.integrate = logical_and(self.integrate,
                                         (~toreset).unsqueeze(-1).expand(self.batch_size,self.N))
    return output

def WinnerTakeAllRefractory(parentclass):
    """
    This superclass copies what has been done with the WinnerTakeAll, and adds
    another shared functionality - the refractoriness of the neurons, once one
    fires. Refractoriness MUST be added via use of the refractory or 
    refractory_variabletimestep superclasses, otherwise is not considered.

    Parameters
    ----------
    parentclass : neurongroup model

    Returns
    -------
    childclass : WTAR_neurongroup model

    """
    tag = "WTAR_"+parentclass.__name__
    kwargs = {"dynamic_class":True,
              "advance_timestep":WTAR_advance_timestep}
    childclass = type(tag,(parentclass,),kwargs)
    return childclass

# @classmethod
def WTAR_advance_timestep(self,local_input):
    output = super(self.__class__, self).advance_timestep(local_input)    
    with no_grad(): # TODO: experimental - might need to review this one
        toreset = self.integrate.any()
        if toreset:
            self.integrate[:] = 0.            
    return output
    
