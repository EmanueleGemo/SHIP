#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:14:42 2023

@author: egemo
"""

# import inspect
from .core import core
from .utils import Dict
from .group import neurongroup,synapsegroup
from torch import (stack,
                   zeros,
                   ones,
                   cat,                   
                   arange,
                   tensor,
                   empty)

class network(core):
    """
    This class is the fundamental piece that assembles the layers into a single
    structure and drives all the inference routines across such structure.
    """
    # 
    
    output = []    
    groups = Dict()
    __stack = []
    __output_idx = []    
    __has_changed = True
    __needs_initializing = True
    
##############################################################################
# class init methods
    def __init__(self,identifier = None):        
        """
        Generator function for the network instance
        
        LOG - it looks like no more tha one network can be present in the 
        workspace - to be corrected.

        Parameters
        ----------
        identifier : any, optional
            Unique identifier of the network. The default is None.

        Returns
        -------
        None.

        """
        # Also here, some of this stuff may be moved up as a class attribute.
        
        self.tag = identifier
        # self.stack = []
        # output params
        # self.output = [] # placeholder for simulation output
        # self.output_idx = False # index along the stack of the output layers
        
        # other params
        # self.has_changed = True
        # self.output_idx = []

        # groups will be stored here
        # self.groups = Dict() # redundant?
        # self.__dict__ = self.groups # allows to seek the network groups via dot notation - LOG to be changed, does not work as intended
        self.groups.clear() # fixes issue by which any new object instance would remember any previous one
        
    def save(self,filename):
        """
        Not yet implemented - this would eventually facilitate the storage of
        the network
        """
        pass
    
    def load(self,filename):
        """
        Not yet implemented - this would eventually facilitate the retrieval of
        the network
        """
        pass
    
    def add(self,obj,tag,*args,**kwargs):
        """
        This function stores new instances as network's layers, and switches
        the network's bool attribute has_changed to True

        Parameters
        ----------
        obj : class
            group class that needs to be stored.
        tag : str
            group unique identifier.
        *args : any
            class init arguments.
        **kwargs : any
            class init keyword arguments.

        Returns
        -------
        None.

        """
        # if tag in dir(self):
        #     raise ValueError('The current object already have an attribute '
        #                     +'named %s (type = %s)'%(tag,
        #                     getattr(self,tag).__class__.__name__)
        #                     +' - the provided tag must be changed')
        # else:  
        #     # # converts tags to obj pointers
        #     # if 'source' in kwargs:
        #     #     if kwargs['source'] in self.groups:
        #     #         kwargs['source'] = self.groups[kwargs['source']]
        #     # if 'target' in kwargs:
        #     #     if kwargs['target'] in self.groups:
        #     #         kwargs['target'] = self.groups[kwargs['target']]
            
        #     self.__groups[tag] = obj(tag,*args, **kwargs)
        #     self.__needs_initializing = True
        #     self.__dict__.update(self.__groups)
        if 'source' in kwargs:
            if kwargs['source'] in self.groups:
                kwargs['source'] = self.groups[kwargs['source']]
        if 'target' in kwargs:
            if kwargs['target'] in self.groups:
                kwargs['target'] = self.groups[kwargs['target']]
        # detect class and calls appropriate adder function
        if neurongroup in obj.__mro__:#inspect.getmro(obj):
            self.add_neurongroup(obj,tag,*args,**kwargs)
        elif synapsegroup in obj.__mro__:#inspect.getmro(obj):
            self.add_synapsegroup(obj,tag,*args,**kwargs)
        setattr(self,'has_changed', True)
    
    def remove(self,tag):
        """
        This function removes a group of the network, if the tag identifier
        is found. It also switches the network's bool attribute has_changed to 
        True

        Parameters
        ----------
        tag : str
            Group unique identifier.

        Returns
        -------
        None.

        """
        if tag in self.groups:
            for gg in self.groups[tag].source.values():
                if tag in gg.target:
                    del gg.target[tag]
            for gg in self.groups[tag].target.values():
                if tag in gg.source:
                    del gg.source[tag]
            del self.groups[tag]
        setattr(self,'has_changed', True)
        
        # if tag in self.__groups.keys():
        #     self.__groups[tag].__remove_from_target()
        #     self.__groups[tag].__remove_from_source()
        #     self.__dict__.pop(tag)
        #     del self.__groups[tag]
        #     # self.__has_changed = True
        #     self.__needs_initializing = True
        #     # for gg in self.__groups[tag].source.values():
        #     #     if tag in gg.target:
        #     #         del gg.target[tag]
        #     # for gg in self.groups[tag].target.values():
        #     #     if tag in gg.source:
        #     #         del gg.source[tag]
        #     # del self.groups[tag]
    
    def add_neurongroup(self,obj,tag,*args,**kwargs):
        if tag in self.groups:
            raise ValueError('A neurongroup object with the same tag already exists.')
        elif tag.find('-') >= 0:
            raise ValueError('Neurongroup tags can not contain the character "-"')
        else:
            self.groups[tag] = obj(tag,*args,**kwargs)
    
    def add_synapsegroup(self,obj,tag,*args,**kwargs):
        if tag in self.groups:
            raise ValueError('A synapsegroup with the same tag already exists.')
        else:
            self.groups[tag] = obj(tag,*args, **kwargs)
        
        
    def set_params(self,tag,**kwargs):
        """
        Function that sets the parameters specified in kwargs, on the group
        identified by its unique tag

        Parameters
        ----------
        tag : str
            Group unique identifier.
        **kwargs : any
            keyword arguments for the group initialization.

        Returns
        -------
        None.

        """
        if tag in self.groups:
           tmp1 = self.groups[tag].arg2InitDict(kwargs)
           if tmp1:
               self.groups[tag].init_variables.update_existing(tmp1)
               self.groups[tag].init_parameters.update_excluding(tmp1,list(self.groups[tag].init_variables.keys()))  
        else:
           raise Exception('group "%s" not found in network object'%tag) 
           
        # # get group
        # if tag in self.__groups.keys():
        #     tmp1 = self.__groups[tag].__arg2InitDict(kwargs)
        #     if tmp1:
        #         self.__groups[tag].init_variables.update_existing(tmp1)
        #         self.__groups[tag].init_parameters.update_excluding(tmp1,list(self.__groups[tag].init_variables.keys()))  
        # else:
        #     raise Exception('Group "%s" not found in network object'%tag) 
            
    def set_monitor(self,*args,**kwargs): ### LOG ### - to fix
        """
        Network setter function for any monitor.
        If naked arguments are passed, the function assumes a str unique 
        identifier and a str attribute identifier. Otherwise, a dict is assumed 
        in which the keys correspond to the groups identifiers, and a list of 
        strings are assumed as attributes.

        Parameters
        ----------
        *args : [str,str]
            Group identifier - Attribute identifier.
        **kwargs : {key:[str]}
            See above.

        Returns
        -------
        None.

        """
        if args: #assuming only one argument is passed
            self.groups[args[0]].set_monitor(args[1])
            if self.groups[args[0]].monitor:
                self.groups[args[0]].run_wrapper = self.groups[args[0]].monitored_step
        else: # a dict is expected
            for k,lst in kwargs.items():
                for i in lst:
                    self.groups[k].set_monitor(i)
                if self.groups[k].monitor:
                    self.groups[k].run_wrapper = self.groups[k].monitored_step
    
    def del_monitor(self,*args,**kwargs): ### LOG ### - to fix
        """
        This function removes the user-defined monitors. 
        If naked arguments are passed, the function assumes a str unique 
        identifier and a str attribute identifier. Otherwise, a dict is assumed 
        in which the keys correspond to the groups identifiers, and a list of 
        strings are assumed as attributes.
        TO DO - the wrapper is used to do both monitoring and apply functionalities
        removing one wreak havoc on the added functionalities, to rework
        

        Parameters
        ----------
        *args : [str,str]
            Group identifier - Attribute identifier.
        **kwargs : {key:[str]}
            See above.

        Returns
        -------
        None.

        """
        if args: #assuming only one argument is passed
            self.groups[args[0]].del_monitor(args[1])
            if self.groups[args[0]].monitor:
                pass
            else: # issue here - advance_timestep might not be the correct wrapper
               self.groups[args[0]].run_wrapper = self.groups[args[0]].advance_timestep
        else: # a dict is expected
            for k,i in kwargs.items():
                self.groups[k].del_monitor(i)
            if self.groups[k].monitor: 
                pass
            else:
                self.groups[k].run_wrapper = self.groups[k].advance_timestep
                
    def format_monitor_results(self): ### LOG ### - to change and to make it provide a dictionary
        """
        Function that, for all monitors, in-place converts the output of the
        monitor (type: list) to more useful torch.tensor values.
        
        LOG - if any more operations are carried out after calling this 
              function, issues will certainly arise
              
        Returns
        -------
        None.

        """
        for gg in self.stack:
            for kk,mm in gg.monitor.items():
                gg.monitor[kk] = stack(mm,dim=1)
                
##############################################################################
# run - init methods        

    def init(self,**kwargs):
        """
        Manual init method for the network class.
        This method performs the following operations:
            1) calls the init function of each of the network group
            2) calls the init_network_structure function that builds the 
               network stack attribute, through which the run function operates
            3) generates the output indices, again useful for the run function
            4) assignes the input handler function to each group, based on the
               network structure
        Before doing so, it also checks if dt - nts - batch_size arguments 
        are passed (at the moment, those can also be passed via core class)
        
        Parameters
        ----------
        **kwargs : any
            all keyword arguments are passed to all the called functions.

        Returns
        -------
        None.

        """
        
        if 'dt' in kwargs:
            core.dt = kwargs['dt']
            del kwargs['dt']
        if 'batch_size' in kwargs:
            core.batch_size = kwargs['batch_size']
            del kwargs['batch_size']
        if 'nts' in kwargs:
            core.nts = kwargs['nts']
            del kwargs['nts']    
        if 'output' in kwargs:
            out = kwargs['output'] 
        else:
            out = []
            
        self.init_groups(**kwargs)
        self.init_network_structure()
        self.init_output_indices(out)
        core.dt_sequence = None
        # self.assign_groups_step_function()
    
    def init_groups(self,**kwargs):
        """
        Handler that calls the init function of each one of the groups.

        Parameters
        ----------
        **kwargs : any
            arguments to be passed to the groups.

        Returns
        -------
        None.

        """
        
        for gg in self.groups.values():
            gg.init(**kwargs)
            # except Exception as e:
            #     raise Exception('Error in group %s. Verify the initialisation parameters. Message: %s'%(g.tag,e))
                # sys.exit('Error in group %s. Verify the initialisation parameters'%(g.tag))
                
    # def assign_groups_step_function(self): <- next version
    #     """
    #     This function attributes a specific step method to each of the 
    #     network groups, based on the network structure and groups' parameters.
    #     Here, we assume the following:
    #         1) the groups' i_chX and o_chX attributes have already been 
    #            processed (those are the input and output indices of the res 
    #            variable respectively) 
    #         2) each group has at most 4 input channels and 4 output channels,
    #            numbered from 0 to 3, and used exclusively in a top up fashion
    #            (e.g. using ch0 first and only then using ch1, and so on)
    #         3) each input channel handler can either fetch one item (if the 
    #            index of such a channel contains one value), or fetch the 
    #            multple items and deliver the SUM (if the index of such channel 
    #            contains multiple values). No other cases are considered.
               
    #     Returns
    #     -------
    #     None.

    #     """
    #     for gg in self.__groups.values():
    #         gg.select_step_function()
    
     
    def init_output_indices(self,output = []):
        """
        This function determines the index (along the network stack variable) 
        corresponding to the network's output, if present. The variable output
        (str) can be passed here, as well as being preemptively stored during 
        the add function call via "is_output = True"

        Parameters
        ----------
        output : str, optional
            Output group's unique identifier. The default is [].
            
        Returns
        -------
        None.

        """
        self.output_idx = []
        self.isanyoutput = False
        if output:
            tmp = ([gg.tag for gg in self.stack]).index(output)
            if tmp:
                self.output_idx = tmp
                self.isanyoutput = True
        else:
            is_output = [gg.is_output for gg in self.stack]
            if any(is_output):
                setattr(self,'output_idx',is_output.index(True))
                self.isanyoutput = True
        if self.isanyoutput:
            if isinstance(self.output_idx,int): # just one output
                setattr(self,'output_hook',[self.stack[self.output_idx]])
            else: # many outputs
                setattr(self,'output_hook',[self.stack[_] for _ in self.output_idx])
        
    def init_network_structure(self, force_init:bool = False):
        """
        This method orders the groups so as to determine the correct temporal
        sequence of the groups' models call. The results is stored as a network
        attribute self.stack (list of groups)
        The function runs only if the network is being changed (which triggers 
        the bool attribute has_changed to True
        
        LOG - manual change of the networks' attributes will not trigger 
        has_changed = True, which negates the method functionality - a 
        force_init bool is introduced so as to avoid this issue.

        Parameters
        ----------
        force_init : bool, optional
            If True - it skips the check of the has_changed attribute. The
            default is False.

        Returns
        -------
        None.

        """
        
        # this method orders the  groups so as to determine the correct
        # group run sequence
        
        if self.has_changed or force_init:
            
            # determine the labels of each group
            tags = [gg.tag for gg in self.groups.values()]            
            n = len(tags)
            
            # thinking of the network as a graph (where each group is a node), 
            # we collect the edges (source-target pairs) and their eventual 
            # delay value            
            pairs = []
            delays = []
            # st: source tag; tt: target tag
            for ii,st in enumerate(tags):
                if hasattr(self.groups[st], 'delay_time'): # <- see if delay
                    d = self.groups[st].delay_time
                else:
                    d = 0 # else set delay to 0
                for tt,gg in self.groups[st].target.items():
                    pairs.append([st,tt]) # <- set delay for any sourcee
                    delays.append(d)
            delays = tensor(delays)
            
            ### SEE DETAILS on get_stack_sequence to find how we do this
            optimal_list = self.get_stack_sequence(pairs,delays,tags)
            
            stack = [self.groups[tt] for tt in optimal_list]
            setattr(self,'stack',stack)
            # we once again go through all elements of the list to fix various bits
            for gg in self.groups.values():
                
            # here we determine the indices of the inputs along the stack
                # input_idx =             
                gg.input_idx = [ii for ii,ss in enumerate(stack) if ss.tag in gg.source]
                
            # here we determine how many inputs each group expects
            
                n = len(gg.source)
                if n == 1:
                    gg.needs_1_input = True
                    gg.needs_more_inputs = False
                    gg.input_idx = gg.input_idx[0] #first element of the list
                elif n>1:
                    gg.needs_1_input = False
                    gg.needs_more_inputs = True
                else:
                    gg.needs_1_input = False
                    gg.needs_more_inputs = False
                    

            # last bit - we recover the Ns/Nt variables for each group
                if gg.is_synapse:
                    gg.Ns = gg.get_Ns() # gg.source.first().N
                    gg.Nt = gg.get_Nt()#gg.target.first().N
                    gg.N = gg.get_N()
                else:
                    gg.Ns = gg.N
                    gg.Nt = gg.N
                        
            setattr(self,'has_changed',False)
                
           
##############################################################################
# run - inference methods
        
    def run(self,*args,**kwargs): 
        """
        This method performs the shallow initialization (set up initial state),
        the inference and the post-processing on the network instance.

        Parameters
        ----------
        *args : any
            arguments to be passed to the network's groups.
        **kwargs : any
            keyword arguments to be passed to the network's groups.

        Returns
        -------
        None.

        """
        
        self.shallow_init(*args,**kwargs) 
        if self.isanyoutput:
            self.run_()
        else:
            self.run_no_out()
        self.post(*args,**kwargs)
        
    def run_multisteps(self,params,*args,**kwargs): 
        """
        general inference function, running the group functions sequentially,
        following the layer stack sequence calculated in init_network_structure
        this version runs according to a set of parameters (dt, nts) that changes during the run
              

        Parameters
        ----------
        params : list
            list or set of [dt,nts] pairs to use during inference
        *args : any
            arguments to be passed to the network's groups.
        **kwargs : any
            keyword arguments to be passed to the network's groups.

        Returns
        -------
        None.

        """
    # 
        self.shallow_init(*args,**kwargs)
        core.dt_sequence = empty(0)
        
        for dt,nts in params:
            core.dt = dt
            core.nts = nts
            core.dt_sequence = cat((core.dt_sequence,ones(nts)*dt),0)
            for gg in self.stack:
                gg.time_dep()

            if self.isanyoutput:
                self.run_()
            else:
                self.run_no_out()                
            
        self.post(*args,**kwargs)
        
    def shallow_init(self,*args,**kwargs):
        """
        Resets the network and its groups to their initial states, according
        to the user specifications.
        NOTE: any input to the network is currently stored into the "input" 
        layers, but the user-provided input argument is passed to such layers 
        here during the shallow init.

        Parameters
        ----------
        *args : any
            Arguments passed to the groups (any input is also passed to the 
            group here).
        **kwargs : any
            Keyword arguments passed to the groups (any input is also passed to
            the group here)..

        Returns
        -------
        None.

        """
        
        # set the output variable to an empty list
        self.output = []
        # res is the black box into which all groups store their outputs
        # (and from which all groups fetch their inputs)
        # here, trivially, we set the results to a set of zero-valued tensors
        # LOG - if multiple outputs/inputs are expected, this strategy does 
        # not work properly. for future development - adjust res so as to force
        # a zero-valued output according to each group. a way might be:
        #
        # for gg in self.stack:
        #     # gg.output = []
        #     gg.set_attributes_from(gg.init_variables) 
        #     gg.time_dep()
        #     gg.set_initial_state(*args,**kwargs)
        #
        #  -->self.res[ii] = someoutput(gg[zero-input])
        #
        # better approach - use default values of groups.
        #
        self.res = [zeros([gg.batch_size,gg.Nt], device = gg.device, dtype = gg.dtype) for gg in self.stack]
        
        for gg in self.stack:
            # use the stored generator functions to assign to the groups'
            # attributes the user-provided values
            gg.set_attributes_from(gg.init_variables) 
            # recalculate any time-dependent parameter
            gg.time_dep()
            # run the user-defined set_initial_state function, that should
            # perform any of the algorithmic steps not enforced automatically
            gg.set_initial_state(*args,**kwargs)            
            # reset monitors here
            for k in gg.monitor.keys():
                gg.monitor[k] = []
        
    def run_(self): 
        """
        Unlike the run method, this one behaves as a "continue", as it does not 
        force an initial state but simulates the temporal dynamics following 
        the initial state already present.

        Returns
        -------
        None.

        """
        
        output = []
        
        for t in range(self.nts):
            
            for ii,gg in enumerate(self.stack):
                # this if-else statement is the best option reduce the overhead
                # due to the local_input dynamically created variable
                if gg.needs_1_input: # send one tensor
                    gg.output = gg.run_wrapper( self.res[gg.input_idx] )
                elif gg.needs_more_inputs: # send sum of input tensors
                    gg.output = gg.run_wrapper( sum([self.res[_] for _ in gg.input_idx]) )
                else:
                    gg.output = gg.run_wrapper()
                self.res[ii] = gg.output   
            # get output here #LOG: at the moment, the code handles only one output
            output.append(self.res[self.output_idx])

        
        if isinstance(self.output,list):
            # create output
            self.output = stack(output,dim=1)            
        else: 
            # if another (tensor) output already exists - concatenate results
            self.output = cat((self.output,stack(output,dim=1)),dim=1)
    
    def run_no_out(self): 
        """
        Same as run_, without checking for outputs.

        Returns
        -------
        None.

        """
        # this function would be a "run_continue", as it does not involve the 
        # determination of the initial state but simulates the temporal dynamics
        # consequent to the initial state, whatever this might be.
        
        # res = [torch.zeros([gg.batch_size,gg.Nt], device = gg.device, dtype = gg.dtype) for gg in self.stack]
        for t in range(self.nts):
            
            for ii,gg in enumerate(self.stack):
                # this if-else statement is the best option reduce the overhead
                # due to the local_input dynamically created variable
                if gg.needs_1_input: # send one tensor
                    gg.output = gg.run_wrapper( self.res[gg.input_idx] )
                elif gg.needs_more_inputs: # send sum of input tensors
                    gg.output = gg.run_wrapper( sum([self.res[_] for _ in gg.input_idx]) )
                else:
                    gg.output = gg.run_wrapper()
                self.res[ii] = gg.output
                
    # def run_1step(self,dt): <-- next version
    #     """
    #     variable dt implementation.
    #     This method runs the network inference for a single timestep.
    #     For each element of the stack, this function:
    #         1) autonomously fetch/calculate the group local_input, according to
    #            predetermined rules
    #         2) yields an output, stored
    #     Parameters
    #     ----------
    #     dt : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
    #     # this function runs a single timestep iteration
    #     for ii,gg in enumerate(self.stack):
    #         self.res = gg.step(dt,self.res)
            
        
            
    #     # if gg.needs_1_input: # send one tensor
    #     #     gg.output = gg.run_wrapper( self.res[gg.input_idx] )
    #     # elif gg.needs_more_inputs: # send sum of input tensors
    #     #     gg.output = gg.run_wrapper( sum([self.res[_] for _ in gg.input_idx]) )
    #     # else:
    #     #     gg.output = gg.run_wrapper()
    #     # self.res[ii] = gg.output 
        
    #     # return res
                
    # def check_timeline():        
    #     pass
    
    def post(self,*args,**kwargs):
        """
        Placeholder for eventual post-processing to be performed after network
        runs.
        
        Parameters
        ----------
        *args : any
            Arguments.
        **kwargs : any
            Keyword arguments.

        Returns
        -------
        None.

        """
        pass
    
    
        
    # def inference(self,*args,dt=0.001,nts=1000,batch_size=1,max_batches=1,**kwargs):
    #     # this method proposes a simplified interface to run the inference,
    #     # automatically managing the data series from a data_generator function
    #     #
    #     # if a data generator function is provided, then the code will run 
    #     # it until completion, or until the maximum number of batches is 
    #     # reached
    #     if 'data_generator' in kwargs:
    #         DG = kwargs['data_generator']
    #         # del kwargs['data_generator']
    #         kwargs['dt'] = dt
    #         kwargs['max_batches'] = max_batches
    #         if not('size' in kwargs):
    #             kwargs['size'] = [batch_size,nts,self.stack[0].N]   
    #         # attempt to rebuild the expected size, might not always work
            
    #         for x_local, y_local in DG(**kwargs):
    #             if x_local.is_sparse:
    #                 x_local = x_local.to_dense()
    #             self.run(x_local,y_local)
                    
    #     # otherwise, the code will run the network without generated inputs,
    #     # possibly including the external input (*args)
    #     else:     
    #         cnt = 0
    #         while cnt<max_batches:
    #             cnt +=1
    #             self.run(*args)
        
    # def train(self,dt=0.001,nts=100,batch_size=32,max_batches=10,epochs = 1,*args,**kwargs):        
    #     # this should be a modified version of inference, in which, after the 
    #     # initial init/check, the code would perform inferences followed by 
    #     # synaptic weights modification and network reset
    #     # likely, the train/reset proper function or class would be passed as 
    #     # argument; a generic reset function/class may be here implemented
    #     pass

##############################################################################
# auxiliary methods
    def get_monitored_results(self):
        """
        Formats the monitor outputs and return a Dict variable
        e.g. monitoring X group for y values:
        data = net.get_monitored_results()
        output -> data.X.y, as a tensor of size [batch_size,nts,N,(N)]

        Returns
        -------
        data : Dict
            Dict structure containing the monitored data pre-formatted for 
            accessibility 
            
        """
        data = Dict()
        for g in self.stack:
            if g.monitor:
                data[g.tag] = Dict()
                for k,i in g.monitor.items():
                    try:
                        data[g.tag][k] = stack(i,dim=1)
                    except:
                        data[g.tag][k] = i
        return data
    
    @staticmethod
    def get_stack_sequence(pairs,delays,tags):
        """
        Static method
        This method calculates the sort order of the groups, based on the 
        provided delay time and pair source-target connections, so as to yield
        the lowest lower-triangular-sum of the corresponding delay-substituted
        adjacency matrix.
        12/06/23
        Slightly different approach - the theory is the same (lowering the 
        lower-triangular-sum (lts) of the delay-substituted adjacency matrix), 
        but this algorithm finds directly the lts values for each possible
        combinations of the pair swaps (source-to-target or viceversa), and only
        then proceeds to find the earliest one (starting from the one having
        the lowest lts sum, and going upwards) that results in a feasible
        network graph. Other details in comments

        Parameters
        ----------
        pairs : list
            list of 2-sized lists of source and target group tags.
        delays : tensor or list
            tensor (or list) having the same size as pairs; finds corresponding
            delay time for each one of the source-target pairs in the pairs variable.
        tags : list
            list containing all the tags of the network's groups.

        Returns
        -------
        optimal_list : list
            sorted list of the groups tags.

        """
        
        # pre-fetching source-only, target-only and unconnected groups,
        # that do not need sorting
        sources = [_[0] for _ in pairs]
        targets = [_[1] for _ in pairs]
        connected = list(set(sources+targets))
        unconnected = [_ for _ in tags if _ not in connected]
        source_only = [_ for _ in connected if _ not in targets]
        target_only = [_ for _ in connected if _ not in sources]
        
        # now dealing with groups that might need to be sorted (source_to_target
        # only groups again do not need sorting, but are easier to find within
        # the loop below)
        tosort = [_ for _ in tags if _ not in unconnected+source_only+target_only]
        if len(tosort)<=1: # one or none to be sorted: the stack sequence is trivial 
            optimal_list = source_only+tosort+target_only+unconnected        
        else:    # here we need to calculate the actual sequence.
            pairsTS = []
            delaysTS= []
            # source_to_target tracks groups that directly link target-only with
            # source-only groups, and of course do not need sorting.
            source_to_target = tosort.copy() 
            for p,d in zip(pairs,delays): # looping through the pairs, to process 
                # only those linkning both source and target that need sorting
                if p[0] in tosort and p[1] in tosort: # <- otherwise ignore
                    pairsTS.append(p)
                    delaysTS.append(d)
                    # removing groups from source-to-target when needed
                    if p[0] in source_to_target:
                        source_to_target.remove(p[0])
                    if p[1] in source_to_target:
                        source_to_target.remove(p[1])
                else:
                    pass
            # we add a small number, eps, to the delays, to avoid issues with 
            # zero-valued delays
            delaysTS = tensor(delaysTS)
            above0 = delaysTS[delaysTS>0]
            if len(above0):
                eps = min(delaysTS[delaysTS>0])/(len(above0)+1)
            else:
                eps = 1
            delaysTS = eps+delaysTS
            # now building a bool matrix tracking all combinations of the pair 
            # swaps
            bits = len(delaysTS)            
            binarymap = arange(2**bits).unsqueeze(-1).bitwise_and(2**arange(bits).unsqueeze(0)).ne(0)
            # calculating a lower-triangular sum "equivalent"; here summing all
            # delay contributions, with the ones on the upper-triangular matrix
            # made negative, and removing an arbitray -1 from the positive locations
            # in the lower-triangular sum (so to properly compute zero-valued delays)
            # lts = ( (delaysTS.unsqueeze(0)*binarymap-delaysTS*binarymap.logical_not_()).sum(dim=1) +
            #        binarymap.sum(dim=-1) )
            lts = ( (delaysTS.unsqueeze(0)*binarymap-delaysTS*binarymap.logical_not_()).sum(dim=1) +
                   binarymap.sum(dim=-1) )
            idx = lts.argsort() # sort the computed sums
            # here, from the fastest to the slowest, checking if the theoretical
            # swaps can lead to a buildable stack sequence
            for ii,localbinary in enumerate(binarymap[idx]):
                
                seq = [] # <- sequence of groups that stricly need sorting
                buildable = True
                        
                sortedpairs = [_ if b else [_[1],_[0]] for _,b in zip(pairsTS,localbinary)]
                for p in sortedpairs:
                    p0 = p[0]
                    p1 = p[1]
                    t0 = p0 in seq 
                    t1 = p1 in seq
                    # here, for each pair to sort, we gradually build the stack
                    # according to the 5 cases: 
                    # 1) not already seen (put both source and target in list)
                    # 2) source already in stack - put the target after
                    # 3) target already in stack - put source before 
                    # 4) seen both, already in order - do nothing
                    # 5) seen both, not in order - try to swap until solving; 
                    #    if clashing with pair rules - not buildable, discard 
                    #    and proceeed with another option
                    if not t0 and not t1: #neither seen before - no prob, put'em there
                        seq.append(p0)
                        if p0!=p1:
                            seq.append(p1)
                    elif not t0 and t1: #0 not in draft - no prob, put it before 1
                        seq.insert(seq.index(p1),p0)
                    elif t0 and not t1: # 1 not in draft - no prob, put it after 0
                        seq.insert(seq.index(p0)+1,p1)
                    else: #checking cases
                        ip0 = seq.index(p0)
                        ip1 = seq.index(p1)
                        dist = ip1-ip0
                        if dist<0: # worst case scenario - 0 and 1 present but swapped
                            go = True
                            while go:
                                if [p1,seq[ip1+1]] in sortedpairs: # can not swap
                                    go = False
                                else:
                                    seq.pop(ip1)
                                    ip1+=1
                                    seq.insert(ip1,p1)
                                    dist+=1
                                    if dist==0: # SORTED
                                        go = False
                            if dist<0:
                                go = True
                                while go:
                                    if [seq[ip0-1],p0] in sortedpairs: # can not swap
                                        go = False
                                    else:
                                        seq.pop(ip0)
                                        ip0-=1
                                        seq.insert(ip0,p0)
                                        dist+=1
                                        if dist==0: # SORTED
                                            go = False
                            if dist<0:
                                buildable = False
                                break   
                if buildable:
                    break
            optimal_list = source_only+source_to_target+seq+target_only+unconnected
        return optimal_list
            