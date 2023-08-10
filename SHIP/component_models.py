#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:18:38 2023

@author: egemo
"""

from .core import core
from .group import synapsegroup,neurongroup
from .component_utils import (LS_time_dep,
                          LIF_time_dep,
                          quick_activator,
                          get_alphas)
from torch import (rand,
                   is_tensor,
                   tensor,
                   ones,
                   clamp,
                   zeros,
                   addmm)

###############################################################################
### reference

class standard_model_scaffold(neurongroup,synapsegroup):
    """
    This class is here solely to serve as a reference for further developments.
    """
    
    variables = {'name_of_the_variable': "default_value"}
    parameters = {'_CanAlsoUseUnderscoresAndFunctions__': rand}
    
    def timedep(self):
        """
        This method precalculates all dependencies from the time-step size
        
        Parameters <-- future version
        ----------
        dt : float 
            time-step assumed for the duration of the inference.
            (this version uses the class property)

        Returns
        -------
        None.

        """
        pass
    
    def set_initial_state(self, *args, **kwargs):
        """
        This method sets the model state in such a way to be capable to iteratively
        run the advance_timestep function, or to simply set an arbitrarily-
        determined initial state more elaborate than the one obtainable with 
        the declaration stage. This method is "optional", in the sense that 
        it may happen that the model does not need it.

        Parameters
        ----------
        *args : any 
        **kwargs : any (keyword arguments).

        Returns
        -------
        None.

        """
        pass
    def advance_timestep(self,local_input):
        """
        this method determines new output and new state from the current state
        and input, as a function of the time-step size (incorporated with the
        time_dep method).

        Parameters
        ----------
        local_input : any
            input for the current time-step used by the group, ideally gathered
            from the group source.

        Returns
        -------
        output : any
            output of the current model, set as an arbitrarily-determined
        """
        return []
    
    
###############################################################################
### synapse models (here, input integration is also performed)

class lS_1o(synapsegroup):
    """
    1st order leaky synapse model, in which the synaptic current immediately 
    ramps up to peak value upon receing a neuronal spike, and in which the 
    synaptic current decreases according to an exponential function.
    This model returns an integral value of the synaptic current per time-step.
    """
    variables = {'_I__': 0} # synapse current <--> internal state
    parameters = {'tau_alpha__': 8e-3, # temporal constant [s]
                  'w__': rand, # synaptic weight
                  'w_scale': 1} # scaling factor

    def time_dep(self):
        self.alphaA,self.alphaB = get_alphas(self.tau_alpha,self.dt)

    def set_initial_state(self, *args, **kwargs):
        if not is_tensor(self.w_scale): # data type checking
            self.w_scale = tensor(self.w_scale)

    def advance_timestep(self,local_input=zeros(1)):
        self.I = self.I*self.alphaA + local_input.unsqueeze(-1).expand(self.batch_size,self.Ns,self.Nt)*self.w*self.w_scale
        return (self.I*self.alphaB).sum(dim=1)

# class synapse_0o(synapsegroup):
#     """
#     zeroth-order synapse (rectangular output) - capped output
#     This model returns an integral value of the synaptic current per time-step.
#     """
#     variables = {'_on__': False, # synapse state <--> internal state
#                   '_timer__': 0} # timer used to determine off state
#     parameters = {'tau_alpha__': 8e-3, # temporal constant [s]
#                   'w__': rand, # synaptic weight
#                   'w_scale': 1} # scaling factor

#     # def time_dep(self):
#     def set_initial_state(self, *args, **kwargs):
#         if not is_tensor(self.w_scale): # data type checking
#             self.w_scale = tensor(self.w_scale)

#     def advance_timestep(self,local_input=zeros(1)):
#         # time evolution
#         self.timer[self.on] = self.timer[self.on]-self.tau_alpha[self.on]
#         self.on[self.timer<=0] = False
#         # perturbation
#         self.on[local_input.unsqueeze(-1).expand(self.batch_size,self.Ns,self.Nt)] = True
#         self.timer[local_input.unsqueeze(-1).expand(self.batch_size,self.Ns,self.Nt)] = self.tau_alpha[local_input.unsqueeze(-1).expand(self.batch_size,self.Ns,self.Nt)]
#         return (self.on*self.w*self.w_scale*self.dt).sum(dim=1)

class lS_2o(synapsegroup):
    """
    2nd order leaky synapse.
    leaky synapse model, in which the synaptic current immediately reaches a 
    peak value upon receing a neuronal spike, and in which the synaptic current
    decreases according to an exponential function.
    This model returns an integral value of the synaptic current per time-step.
    """
    variables = {'_I1__': 0,
                 '_I2__': 0}# synapse current contributions <--> internal states
    parameters = {'tau_alpha1__': 8e-3, # temporal constant [s]
                  'tau_alpha2__': 4e-3,                  
                  'w__': rand, # synaptic weight
                  'w_scale': 1} # scaling factor

    def time_dep(self):
       self.alpha1A,self.alpha1B = get_alphas(self.tau_alpha1,self.dt)
       self.alpha2A,self.alpha2B = get_alphas(self.tau_alpha2,self.dt)

    def set_initial_state(self, *args, **kwargs):
        if not is_tensor(self.w_scale): # data type checking
            self.w_scale = tensor(self.w_scale)

    def advance_timestep(self,local_input=zeros(1)):
        local_input = local_input.unsqueeze(-1).expand(self.batch_size,self.Ns,self.Nt)*self.w*self.w_scale
        self.I1 = self.I1*self.alpha1A + local_input
        self.I2 = self.I2*self.alpha2A - local_input
        
        return (self.I1*self.alpha1B+self.I2*self.alpha2B).sum(dim=1)
    
class lS_FZenke(synapsegroup):
    """
    This model has been taken from 
    https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
    Unlike the previous synapse models, this one calculates the temporal dynamics
    as applied to the sum of the incoming pulses. Consequences:
    1) removing a dimension from the state tensor;
    2) faster calculation;
    3) loses part of the temporal dynamics information.
    Differently from Zenke implementation, this model yields the integral of the
    output current, for consistency with the other models.
    """

    variables = {'_I_': 0}
    parameters = {'tau_alpha_': 8e-3,
                  'w__': rand,
                  'w_scale': 1}
    
    # override the function determining the number of devices
    def get_N(self):
        return self.Nt 
    
    # def time_dep(self):
    # #     LS_time_dep(self)
    time_dep = LS_time_dep
    
    def advance_timestep(self,local_input):  
        # self.I = self.I*self.alpha + einsum("bi,io->bo",(local_input.type(self.dtype),self.w*self.w_scale))
        # self.I = self.I*self.alpha + mm(local_input.type(self.dtype),self.w*self.w_scale)
        self.I = addmm(self.I*self.alpha,local_input.type(self.dtype),self.w,alpha=self.w_scale)
        return self.I
    
###############################################################################
### neuron models

class liN(neurongroup):
    """
    Simplest Leaky-Integrate Neuron model, does NOT fire.
    The rest potential is hard-coded to 0.
    The membrane potential is also clamped to 0.
    The output is the membrane potential (non-integral value)
    """
    
    variables = {'_u_': 0} # neuron membrane potential <--> internal state
    parameters = {'tau_beta_': 1e-3} # temporal constant [s]
    
    # quickly set some of the required functions:
    time_dep = LIF_time_dep    

    def advance_timestep(self,local_input=0):
        self.u = clamp(self.u*self.beta+local_input,min=0)
        return self.u
    
class lifN(neurongroup):
    """
    Simplest Leaky-Integrate and Fire model.
    Rest/reset potential are hard-coded to 0.
    The integration functionality is enabled by a tensorial value "integrate"
    """
    
    variables = {'_u_': 0} # membrane potential <-- internal state
    parameters = {'thr': 1, # threshold potential - the neuron fires once u overcomes its value
                  'tau_beta_': 1e-3} # temporal constant [s]
    
    # quickly set some of the required functions:
    activator = quick_activator
    time_dep = LIF_time_dep    
    # def time_dep(self): <- alternative statement method
    #     LIF_time_dep(self)
    
    def set_initial_state(self,*args,**kwargs):        
        self.integrate = ones(self.u.shape,dtype = bool)# ~self.activator(self.u-self.thr).detach().bool()
    
    def advance_timestep(self,local_input=0):
        self.u = self.integrate*clamp(self.u*self.beta+local_input,min=0)
        spikes = self.activator(self.u-self.thr)
        self.integrate = ~spikes.detach().bool() 
        return spikes
    
class lifN_b(lifN):
    """
    This model expresses Rest and Reset potential with one value.
    The integration functionality is enabled by a tensorial value "integrate"
    """
    
    variables = {'_u_': 0} # neuron membrane potential <--> internal state
    parameters = {'u0': 0, # neuron rest/reset membrane potential;  
                  'thr': 1, # threshold potential - the neuron fires once u overcomes its value
                  'tau_beta_': 1e-3} # temporal constant [s]

    def advance_timestep(self,local_input=0):
        self.u = self.u0+ self.integrate*clamp((self.u-self.u0)*self.beta+local_input,min=-self.u0)
        spikes = self.activator(self.u-self.thr)
        self.integrate = ~spikes.detach().bool() # <-- refractory init
        return spikes

class lifN_c(lifN):
    """
    This model separates rest and reset potential variables
    """
    
    parameters = {'u0': 0, # neuron rest membrane potential; 
                  'ur': 0, # neuron reset membrane potential; 
                  'thr': 1, # threshold potential - the neuron fires once u overcomes its value
                  'tau_beta_': 1e-3} # temporal constant [s]
    def set_initial_state(self,*args,**kwargs):        
        self.integrate = ones(self.u.shape,dtype = bool)
        self.du = self.u-self.u0
        self.thr_u0 = self.thr-self.u0
        
    def advance_timestep(self,local_input=0):
        self.du = self.integrate*clamp(self.du*self.beta+local_input,min=-self.u0)
        spikes = self.activator(self.du-self.thr_u0)
        self.integrate = ~spikes.detach().bool()
        # self.u = self.du*self.integrate+self.u0 <- deleted, only cosmetic
        self.u = self.du+self.u0
        return spikes

###############################################################################
### input/autonomous models

class inputN(neurongroup):
    """
    Simple, but effective, Input Neuron model.
    During the init functiion it stores the first provided argument.
    During inference, it returns a slice of the stored argument.
    """
    
    variables = {}
    parameters = {}

    # simply transfers the input to the output, provided in a args[0] form
    def set_initial_state(self,*args):
        # using this function just to store the argument
        self.input = args[0]
        self.__t = -1                    
    def advance_timestep(self,*args): # the *args would just prevent issue if any further input is accidentally provided
        self.__t += 1
        return self.input[:,self.__t,:]
    
class list_inputN(inputN):
    """
    Provides the input functionality for arguments in list datatype.
    It also permit to modify the number of timesteps according to the longest
    of the provided inputs
    CAREFUL - if not using the adaptive_nts functionality, and a list with
    inputs of size S<nts is provided, the algorithm will likely throw an error.
    """
    variables = {}
    parameters = {'adaptive_nts': True}

    # simply transfers the input to the output, provided in a args[0] form
    def set_initial_state(self,*args,**kwargs):
        # using this function just to store the argument
        
        tt = [len(aa) for aa in args[0]]  
        if self.adaptive_nts:            
            mt = tensor(tt).max().item()
            core.nts = mt
        self.input = zeros(self.batch_size,self.nts,self.N,dtype = bool)
        for ii,aa in enumerate(args[0]):
            self.input[ii,:tt[ii],:aa.shape[1]] = aa
            
        self.__t = -1 # reset timer

    def advance_timestep(self,*args): # the *args would just prevent issue if any further input is accidentally provided
        self.__t += 1
        return self.input[:,self.__t,:]  
    
class poissonN(neurongroup):
    """
    Simple implementation of a self-spiking Poisson Neuron model.
    """
    
    variables = {}
    parameters = {'rate':100}

    def time_dep(self):
        # probability to find a spike per timestep 
        self.p = self.rate*self.dt
    # def set_initial_state(self,*args):
        # ignores any input, and determines the output as follows:
        # self.randomspikes = rand([self.batch_size,self.nts,self.N])<self.p
        # self.__t = -1
    def advance_timestep(self):
        # self.__t += 1
        # return self.randomspikes[:,self.__t,:]
        return rand([self.batch_size,self.N])<self.p#[dt] <-- next version
