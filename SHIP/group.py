#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:00:03 2023

@author: egemo
"""

from .core import core
from .utils import Dict
from torch import (float as torchfloat,
                  zeros,
                  ones,
                  rand,
                  tensor,
                  arange,
                  stack,
                  bool as torchbool,
                  meshgrid)
import matplotlib.pyplot as plt

class group(core):
    """
    The class group corrals all (hierarchically identical) components 
    identifiable withina network. In general terms, this definition includes
    either neuron layers and synapse sets (connecting the same source and 
    target layers), but may also define additional devices and functionalities 
    not strictly adhering to the conventional layer definition, yet 
    fulfilling the broad definition above (e.g. layers of post-neuron 
    processors, dendritic integrators, learning rules circuit blocks, etc.)
    """
##############################################################################
# class init methods
    variables = {}
    parameters = {}
    
    def __init__(self, tag, N = 1, source = {}, target = {}, dtype = torchfloat, is_output = False, is_synapse = False, is_neuron = True, **kwargs):
        """
        Generator function for the generic group class.

        Parameters
        ----------
        tag : str
            unique (string) identifier for the group instance.
        N : int, optional
            Number of unique devices of the group instance. The default is 1.
        source : str, optional
            Identifier of the group layer from which the current instance 
            obtains its input. The default is [].
        target : str, optional
            Identifier of the group layer to which the current instance 
            would send its output. The default is [].
        dtype : data type, optional
            Generic data type for the group instance tensor.
            The default is torch.float.
        is_output : bool, optional
            If True, the network identifies its output as the current instance
            output. The default is False.
        is_synapse : bool, optional
            If True, the network identifies the current group instance as a
            synapse group. The default is False.
        is_neuron : bool, optional
            If True, the network identifies the current group instance as a
            neuron group. The default is False.
        **kwargs : any type
            Additional arguments for the current instance.

        Returns
        -------
        None.

        """
        # placeholders for now
        # self.variables = {} 
        # self.parameters = {}
        self.monitor = Dict() # monitor is determined by the user via use of the set_monitor function
        self.source = Dict()
        self.target = Dict()        
        
        ### dynamic init from here onward
        #
        self.tag = tag # unique identifier
        #
        # N, Nt, Ns are the number of elements in the group /target group /source group
        # those aredynamically determined for synapses, but we may need to create 
        # placeholders for further init functions anyway)
        self.N = N 
        self.Ns = N
        self.Nt = N
        # 
        self.dtype = dtype
        #
        self.is_synapse = is_synapse
        self.is_neuron = is_neuron
        self.is_output = is_output
        #
        # connecting vicinal groups to the current one, if provided 
        if target:  
            self.target[target.tag] = target
            for gg in self.target.values():
                gg.source[self.tag] = self    
        if source: 
            self.source[source.tag] = source
            for gg in self.source.values():
                gg.target[self.tag] = self            
        #                   
        # now setting variables, parameters and functions:
        self.needs_1_input = False
        self.needs_more_inputs = False
        self.input_idx = []
        #
        # lazy and inefficient method but it works
        self.init_variables = self.arg2InitDict(self.variables)
        self.init_parameters = self.arg2InitDict(self.parameters)   
        tmp1 = self.arg2InitDict(kwargs)
        if tmp1:
            self.init_variables.update_existing(tmp1)
            self.init_parameters.update_excluding(tmp1,list(self.init_variables.keys()))
        #
        # here I should define the standard wrapper function (timestep function)
        #
        # clarification: the idea of calling a wrapper function solves the need
        # to set different run functions when auxilliary ops needs to be performed,
        # without the direct contribution of the user.
        # in this case, this strategy provides a mean to systematically store the
        # variables without much effort, as calling for set_monitor replaces
        # the wrapper function with a monitored_advance_timestep function.
        self.run_wrapper = self.advance_timestep
                
##############################################################################
# initialisation methods          
            
    def init(self):
        """
        Manual call to the init function that tackles the layer parameters and
        functions. It calls for the initialization of either self.parameters 
        and time-dependent parameters

        Returns
        -------
        None.

        """
        # this function would be called to initialise the group params and funs
        self.set_attributes_from(self.init_parameters)           
        self.time_dep() # init time-dependent parameters
    
    def set_attributes_from(self,arg):
        """
        Helper that stores the results of the functions contained in the arg dict.
        This function is usually called during the initialization.
        
        Parameters
        ----------
        arg : dict or Dict
            dictionary containing the functions to be run so to obtain an output
            that is stored as a class attribute.
        Returns
        -------
        None.

        """
        
        for k,f in arg.items():
            setattr(self,k,f())
    
    
    def set_initial_output(self):
        """
        Merely initializes the output of a group, would be superseded in future

        Returns
        -------
        None.

        """
        
        self.output = zeros([self.batch_size,self.Nt],dtype = self.dtype, device = self.device)
        
    
    # def check_channels(self):  <- next version
    #     """
    #     This method initializes the channel index variables (i_chX and o_chX,
    #     with X comprised between 0 and 3 - hard coded, no more channels)

    #     Returns
    #     -------
    #     None.

    #     """
        
        
        
    # def select_step_function(self):  <- next version
    #     """
    #     Method that attributes a private method to the step function, which 
    #     handles i) input fetching ii) model evaluation iii) output storing

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     string = "__step_"
    #     if self.i_ch0:
    #         string = string+"0"
    #         if len(self.i_ch0)>1:
    #             self.ih0 = getattr(self, "_"+self.__class__.__name__+"__fetch_and_sum")
    #         else:
    #             self.ih0 = getattr(self, "_"+self.__class__.__name__+"__fetch")
    #         if self.i_ch1:
    #             string = string+"1"
    #             if len(self.i_ch1)>1:
    #                 self.ih1 = getattr(self, "_"+self.__class__.__name__+"__fetch_and_sum")
    #             else:
    #                 self.ih1 = getattr(self, "_"+self.__class__.__name__+"__fetch")
    #             if self.i_ch2:
    #                 string = string+"2"
    #                 if len(self.i_ch2)>1:
    #                     self.ih2 = getattr(self, "_"+self.__class__.__name__+"__fetch_and_sum")
    #                 else:
    #                     self.ih2 = getattr(self, "_"+self.__class__.__name__+"__fetch")
    #                 if self.i_ch3:
    #                     string = string+"3"
    #                     if len(self.i_ch2)>1:
    #                         self.ih2 = getattr(self, "_"+self.__class__.__name__+"__fetch_and_sum")
    #                     else:
    #                         self.ih2 = getattr(self, "_"+self.__class__.__name__+"__fetch")
    #     else:
    #         string = string+"n"
        
    #     string = string+"_0" # output on channel0 is always expected
                        
    #     if self.o_ch1:
    #         string = string+"1"
    #         if self.o_ch2:
    #             string = string+"2"
    #             if self.o_ch3:
    #                 string = string+"3"
                    
    #     self.step = getattr(self, "_"+self.__class__.__name__+string)
        
            
##############################################################################
# user-defined functions that determine initial data handling and simulation results
# those would eventually be the functions that the end-user would need to specify

    def set_initial_state(self,*args,**kwargs):
        """        
        This function handles any input and perform further steps other than
        the automated ones, to determine a suitable initial condition of the 
        group instance vars/params. Here the function is a placeholder.
        
        Parameters
        ----------
        *args : any type
        **kwargs : any type

        Returns
        -------
        None.

        """        
        pass
           
    def time_dep(self): # user-defined method
        """        
        This function is specifically set up to calculate the delta_t-dependent 
        parameters.
        
        Parameters
        ----------
        *args : any type
        **kwargs : any type
    
        Returns
        -------
        None.
    
        """ 
        pass
    
    def advance_timestep(self): #(self,dt): <- next version
        """
        his function would be the actual device model.
        It determines the model variables evolution as a function of the 
        timestep size dt

        Parameters
        ----------
        dt : yet to be specified
            timestep size (or index of an internal variable).

        Returns
        -------
        None here, but a local output is expected in Children classes

        """
        # pass - replaced with a return [], for test purposes
        return []

# ##############################################################################
# step function - function that automatically handles the io and temporal advancement
# of the object. Here I list a few combinations that hopefully would cover all
# cases, and whose selection is determined automatically (depending on class and
# user-provided data). Future development might vary those few details and possibly
# generate a step-function dinamically without impacting performance.

    # def step(self,res):  <- next version
    #     """
    #     Step method - function that automatically handles the io and temporal 
    #     advancement of the object.
    #     Placeholder - will eventually be assigned to one of the following 
    #     functions, tagged via number of employed input-output channels.

    #     Parameters
    #     ----------
    #     res : list (or dict?)
    #          group outputs.

    #     Returns
    #     -------
    #     res : list (or dict?)
    #          group outputs.

    #     """
    #     pass
    
    # # input handlers (only fetch and sum are here assumed)
    # @staticmethod
    # def __fetch_and_sum(res,idx):
    #     return sum([res[_] for _ in idx])
    # @staticmethod
    # def __fetch(res,idx):
    #     return res[idx]

    # # returns one item to channel 0
    # def __step_n_0(self,res):
    #     res[self.o_ch0] = self.run_wrapper()        
    # def __step_0_0(self,res):
    #     res[self.o_ch0] = self.run_wrapper(self.ih0(res,self.i_ch0))        
    # def __step_01_0(self,res):
    #     res[self.o_ch0] = self.run_wrapper(self.ih0(res,self.i_ch0),
    #                                        self.ih1(res,self.i_ch1))
    # def __step_012_0(self,res):
    #     res[self.o_ch0] = self.run_wrapper(self.ih0(res,self.i_ch0),
    #                                        self.ih1(res,self.i_ch1),
    #                                        self.ih2(res,self.i_ch2))
    # def __step_0123_0(self,res):
    #     res[self.o_ch0] = self.run_wrapper(self.ih0(res,self.i_ch0),
    #                                        self.ih1(res,self.i_ch1),
    #                                        self.ih2(res,self.i_ch2),
    #                                        self.ih3(res,self.i_ch3))        
    # # returns one item to channel 0 and channel 1
    # def __step_n_01(self,res):
    #     res[self.o_ch0],res[self.o_ch1] = *self.run_wrapper()
        
    # def __step_0_01(self,res):
    #     res[self.o_ch0],res[self.o_ch1] = *self.run_wrapper(
    #         self.ih0(res,self.i_ch0))        
    # def __step_01_01(self,res):
    #     res[self.o_ch0],res[self.o_ch1] = *self.run_wrapper(
    #         self.ih0(res,self.i_ch0),
    #         self.ih1(res,self.i_ch1))
    # def __step_012_01(self,res):
    #     res[self.o_ch0],res[self.o_ch1] = *self.run_wrapper(
    #         self.ih0(res,self.i_ch0),
    #         self.ih1(res,self.i_ch1),
    #         self.ih2(res,self.i_ch2))
    # def __step_0123_01(self,res):
    #     res[self.o_ch0],res[self.o_ch1] = *self.run_wrapper(
    #         self.ih0(res,self.i_ch0),
    #         self.ih1(res,self.i_ch1),
    #         self.ih2(res,self.i_ch2),
    #         self.ih3(res,self.i_ch3))        
    # # returns one item to channel 0, channel 1 and channel 2
    # def __step_n_012(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2] = *self.run_wrapper()
        
    # def __step_0_012(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2] = *self.run_wrapper(
    #         self.ih0(res,self.i_ch0))        
    # def __step_01_012(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2] = *self.run_wrapper(
    #         self.ih0(res,self.i_ch0),
    #         self.ih1(res,self.i_ch1))
    # def __step_012_012(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2] = *self.run_wrapper(
    #         self.ih0(res,self.i_ch0),
    #         self.ih1(res,self.i_ch1),
    #         self.ih2(res,self.i_ch2))
    # def __step_0123_012(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2] = *self.run_wrapper(
    #         self.ih0(res,self.i_ch0),
    #         self.ih1(res,self.i_ch1),
    #         self.ih2(res,self.i_ch2),
    #         self.ih3(res,self.i_ch3))
    # # returns one item to channel 0, channel 1, channel 2 and channel 3:
    # def __step_n_0123(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2],
    #     res[self.o_ch3] = *self.run_wrapper()
        
    # def __step_0_0123(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2],
    #     res[self.o_ch3] = *self.run_wrapper(self.ih0(res,self.i_ch0))        
    # def __step_01_0123(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2],
    #     res[self.o_ch3] = *self.run_wrapper(self.ih0(res,self.i_ch0),
    #                                         self.ih1(res,self.i_ch1))
    # def __step_012_0123(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2],
    #     res[self.o_ch3] = *self.run_wrapper(self.ih0(res,self.i_ch0),
    #                                         self.ih1(res,self.i_ch1),
    #                                         self.ih2(res,self.i_ch2)) 
    # def step_0123_0123(self,res):
    #     res[self.o_ch0],res[self.o_ch1],res[self.o_ch2],
    #     res[self.o_ch3] = *self.run_wrapper(self.ih0(res,self.i_ch0),
    #                                         self.ih1(res,self.i_ch1),
    #                                         self.ih2(res,self.i_ch2),
    #                                         self.ih3(res,self.i_ch3))
        
###############################################################################
# monitor methods         

    def monitored_step(self,*arg):
        """        
        This function supersedes the default wrapper, catches its output and
        uses it as a function output, but not before quering for a set of
        user-determined attributes that this function stores in a separate
        local attribute, for it to be accessed after inference (and which would
        otherwise be lost)
        
        Parameters
        ----------
        arg: any argument accepted by the local instance class
            class instance inputs
    
        Returns
        -------
            class instance output
    
        """
        self.output = self.advance_timestep(*arg) 
        # must store output here, otherwise the following loop would not see it
        for k,v in self.monitor.items():
            v.append(self.__dict__[k])
        return self.output
    
##############################################################################
# auxiliary mehtods and helpers

    def arg2InitDict(self,arg = {}): 
        """
        Along with the following functions (find_vars_from_Dict, and set_ki),
        this function determines the key-items couples to be processed and 
        stored as attributes of the class instance. Implicit rule: all 
        variables are to be considered "as is", in the sense that no assertion
        is implied, and whatever is provided by the user will be taken as an
        immutable parameter.
        Exceptions:
           1) numpy.array data is converted to torch.tensor data
           2) all data is, in fact, converted to a generator function, which
              may or may not take as arguments the batch_size, the source and
              the target dimensions, or the current group dimension (self.N)
              
              The rule by which this is determined is the following.
              
              Assuming a key named TEST_KEY, the function looks for the char
              underscore (_, or ALT+95) before and after the key. If any is 
              found at the beginning of the key string (e.g. _TEST_KEY), the 
              generator function will use the batch_size as an argument. 
              
              If an underscore is found at the end of the key as in TEST_KEY_
              (or_TEST_KEY_) the generator function will use self.N as an 
              argument (or self.batch_size and self.N)
              
              If two underscores are found at the end of the key, as in 
              TEST_KEY__ (_TEST_KEY__) the generator function will use self.Ns
              and self.Nt as arguments (self.batch_size, self.Ns, self.Nt)
              
              LOG -- self.Ns and self.Nt may not be accessed directly in future
              and in stead of those specific getter functions may be adopted.
              
        Parameters
        ----------
        arg : dict or Dict, optional
            dictionary of variables to be processed and stored as attributes
            (generator function). The default is {}.

        Returns
        -------
        d : Dict
            dictionary of generator functions.

        """
        
        d = Dict()
        for k,i in arg.items():
            found = self.find_vars_from_Dict(k,i)
            d[found[0]] = self.set_ki(*found)
        return d
    
    @staticmethod
    def find_vars_from_Dict(k,i):
        """
        Gets user-provided key and item, and returns a key stripped from 
        unnecessary underscores along with the item and boolean values 
        identifying the requirements of batch_size or self.N/self.Ns/self.Nt as 
        arguments for the generator function.

        Parameters
        ----------
        k : str
            user-provided key.
        i : any
            user-provided argument to be stored as class instance attribute 
            (via generator function, if not a function itself).

        Returns
        -------
        k : str
            key stripped of initial and final underscores.
        i : any
            same as Parameter.
        b : bool
            True if batch_size dependency is found.
        n1 : bool
            True if self.N dependency is found.
        n2 : bool
            True if self.Ns, self.Nt dependency is found.

        """
        
        # getting/stripping underscores off key
        if k[0] == '_': # insert batch size as param on function
            k = k[1:] 
            b = True
        else:
            b = False
            
        if k[-1] == '_': # possibly use N as param on function
            k = k[:-1]
            n1 = True
            if k[-1] == '_': # use Ns, Nt as params on function
                k = k[:-1]
                n2 = True
            else:
                n2 = False
        else:
            n1 = False
            n2 = False
        return k,i,b,n1,n2
    
    def set_ki(self,k,i,b,n1,n2):
        """
        This function returns a generator function based on the user provided
        inputs

        Parameters
        ----------
        k : str
            attribute key (unused).
        i : any
            user-provided item.
        b : bool
            True if batch_size dependency is necessary.
        n1 : bool
            True if self.N dependency is necessary.
        n2 : bool
            True if self.Ns, self.Nt dependency is necessary.

        Returns
        -------
        function
            generator function based on the provided specification (b,n1,n2) 
            and item i.

        """
        
        if (type(i) is type(lambda _:_)) | (type(i) is type(rand)): # item is a function
            if b:
                if n1:
                    if n2:
                        return lambda : i(self.batch_size,self.Ns,self.Nt)
                    else:
                        return lambda : i(self.batch_size,self.N)
                else:
                    return lambda : i(self.batch_size)
            else:
                if n1:
                    if n2:
                        return lambda : i(self.Ns,self.Nt)
                    else:
                        return lambda : i(self.N)
                else:
                    return i
                    
        else: # assuming a numerical value of the item
            # only exception to the "no data handling" hidden rule: 
            # convert numpy arrays to torch tensor
            if type(i) == 'numpy.ndarray': 
                i = tensor(i)
            
            if b:
                if n1:
                    if n2:
                        return lambda : i*ones(self.batch_size,self.Ns,self.Nt)
                    else:
                        return lambda : i*ones(self.batch_size,self.N)
                else:
                    return lambda : i*ones(self.batch_size)
            else:
                if n1:
                    if n2:
                        return lambda : i*ones(self.Ns,self.Nt)
                    else:
                        return lambda : i*ones(self.N)
                else:
                    return lambda : i
            
    def set_monitor(self,var_name):
        """
        This function initializes the monitor attribute.
        It get name of arguments of advance_timestep function
        Here, we assume that a refractory/delayed class is created dynamically, 
        and thus the argument list needs to be retrieved from the class parent.
        Future development may store the variables at the __init__ stage

        Parameters
        ----------
        var_name : str
            Identifier of the attribute to be monitored during inference.

        Returns
        -------
        None.

        """
        
        
        if var_name in self.init_variables or var_name in self.init_parameters:
            self.monitor[var_name] = []
        elif var_name == "output":
            self.monitor['output'] = []
        else:
            print('Issue in monitor setup: %s not stored in %s data class (%s: variables = %s, parameters = %s)'%(var_name,self.tag,self,list(self.init_variables.keys()),list(self.init_parameters.keys())))    

    def del_monitor(self,var_name):
        """
        Analogally to set_monitor, this function destroys the monitor of the 
        var_name attribute (if present)

        Parameters
        ----------
        var_name : str
            Attribute that is not required to be monitored anymore.

        Returns
        -------
        None.

        """
        
        if not isinstance(var_name,list):
            var_name = [var_name]
        for vv in var_name:
            if vv in self.monitor:
                del self.monitor[vv]
            else:
                print('Variable %s not found in current monitor (current list: %s)'%(var_name,list(self.monitored.keys())))
    
    @staticmethod    
    def strip_keys(d):
        """
        Helper that removes the underscores from sting d

        Parameters
        ----------
        d : str

        Returns
        -------
        dk : str
            same as d but without inital/final underscores.

        """
        
        
        dk_ = list(d.keys())
        dk = []
        
        # strip _s off keys
        for kk in dk_:
            tmp = kk
            if tmp[0] == '_':
                tmp = tmp[1:]
            if tmp[0] == '_':
                tmp = tmp[1:]
            if tmp[-1] == '_':
                tmp = tmp[:-1]
            if tmp[-1] == '_':
                tmp = tmp[:-1]            
            dk.append(tmp)
            
        return dk
    
    def plot_batch_monitor(self, var, batch = 0, dts = None, indices_range = None, mksize = 10, **kwargs):      
        """
        Simple function that plots a monitored attribute (var) along the
        temporal axis, for a given batch (0 by default)

        Parameters
        ----------
        var : str
            attribute that, if monitored, is being plotted.
        batch : int, optional
            value of the index pointing to the batch of interest. The default is 0.
        dts : int, optional
            length of the temporal axis (in timesteps). If None, all the 
            temporal axis is evaluated. The default is None.
        indices_range : torch.tensor, optional
            Device indices used to limit the eventual plot. If None,
            all of the indices are plotted. The default is None.
        mksize : float, optional
            Plot marker size. The default is 10.
        **kwargs : any
            matplotlib.pyplot.figure() arguments.

        Returns
        -------
        None.

        """
        if var in self.monitor:
            if len(self.monitor[var])>0:
                plt.figure(**kwargs)
                
                if (dts is None) or dts>len(self.monitor[var]):
                    dts = len(self.monitor[var])
                if indices_range is None:
                    indices_range = arange(self.Nt)
                to_plot = stack(self.monitor[var],dim=1).detach()[batch][:dts,indices_range]
                
                if to_plot.dtype == torchbool:
                    plt.scatter(self.timeline(dts).expand([len(indices_range),dts]).t()[to_plot],
                                indices_range.expand(dts,len(indices_range))[to_plot],s=mksize)
                    plt.ylabel('index')
                else:
                    plt.plot(self.timeline(dts),to_plot,'-')
                    plt.ylabel(self.tag+' - '+var)
                
                # plt.ylim(0, len(indices_range))
                # plt.gca().set_yticklabels(indices_range.numpy())
                plt.xlabel('Time [s]')
                
    
##############################################################################
# neuron and synapse subclass

class neurongroup(group):
    """
    Generic neuron class, child of the generic group class. 
    At the current stage, the difference lays in the definition of Ns and Nt,
    which by default are both stored as N (neurongroups drive the device number
    definition)
    
    TO DO - neuron/synapse groups will eventually be removed, so to use only 
    "static N" and "dependent N" groups (will need to define everything and 
     check for backcompatibility with the trainer)
    """
    def __init__(self,tag, N = 0, source = [], target = [], dtype = torchfloat, *args, **kwargs):
        super().__init__(tag, N, source, target, 
                         is_synapse = False,
                         is_neuron = True,
                         # input_handler = "integrate",
                         dtype = dtype,
                         **kwargs)
        
        if 'N' in kwargs.keys(): # should always be?
            self.Ns = self.N
            self.Nt = self.N

class synapsegroup(group):
    """
    Generic synapse class, child of the generic group class. 
    At the current stage, the difference lays in the definition of N, Ns and Nt,
    which by default are dependent on the source and target N values.
    It also defines new functions as getter for N, Ns and Nt, which may become 
    useful when trying to reduce the size of the device by use of hidden values
    of N, Ns, Nt (and mapping function?)that do not interfere with an otherwise
    working framework.
    
    TO DO - remove (see neurongroup)
    """
    def __init__(self,tag, N = 0, source = [], target = [], dtype = torchfloat, *args, **kwargs):
        # shared params
        self.w_scale = 1
        # shared init procedure
        super().__init__(tag, N, source, target, 
                         is_synapse = True,
                         is_neuron = False,
                         # input_handler = "fetch",
                         dtype = dtype, 
                         **kwargs)
        
        if self.source and self.target:
            try:
                self.Ns = self.get_Ns()
                self.Nt = self.get_Nt()
                self.N = self.get_N()           
                # self.recheck = False
            except:
                pass
            
    def get_Ns(self):
        """
        Source N getter
        (obtained quering the first of the instance's sources)

        Returns
        -------
        int
            number of devices of the group source.

        """
        return self.source[next(iter(self.source))].N 
    def get_Nt(self):
        """
        Along the lines of get_Ns, this is the Target N getter

        Returns
        -------
        int
            number of devices of the group target.

        """
        return self.target[next(iter(self.target))].N 
    def get_N(self):
        """
        N getter (mayy be modified by more sophisticated classes)

        Returns
        -------
        int
            number of devices of the group instance.

        """
        
        return self.Ns*self.Nt 
        
    def visualize_connectivity(self,w = None, mksize = 100, **kwargs):
        """
        Simple function that rapidly plots the synaptic connection between 
        the group's source and target

        Parameters
        ----------
        w : tensor, optional
            By default, w is the weight parameter. The default is None - if 
            None, the function will use self.w 
        mksize : int, optional
            matplotlib.pyplot.scatter() marker size. The default is 100.
        **kwargs : any
            matplotlib.pyplot.figure() arguments.

        Returns
        -------
        None.

        """
        
        if w is None:
            if hasattr(self,'w'):
                w = self.w
        if w is None:
            print('weight parameter not present')
        else:                
            Ns,Nt = w.shape
            plt.figure(**kwargs)
            mask = w>=0
            x,y = meshgrid(arange(Ns), arange(Nt))
            plt.scatter(x,y, (mask*w*mksize).detach().numpy(),'b') ### LOG 08/02/2022, added detach numpy bit
            plt.scatter(x,y, ((~mask)*(-w)*mksize).detach().numpy(),'r')
            plt.xlim(-1, Ns+1)
            plt.ylim(-1, Nt+1)
            plt.xlabel('Source neuron index')
            plt.ylabel('Target neuron index')
            