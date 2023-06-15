# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:50:34 2023

@author: Em
"""
from torch import (zeros_like,
                   abs as t_abs)
from torch.optim import Adam
from torch.nn import (LogSoftmax,
                      NLLLoss)
from torch.autograd import Function

class trainer():
    """
    This simple class is the backbone consenting to build a proper class that
    can be the used to rapidly train a network object by way of the PyTorch routines.
    As with the components' models, we define a few main methods that need to 
    be amended for the trainer to be functional:
        __init__ (private) -> interfaces the net object with the trainer object. 
        init -> pre-set the trainer in an initial state where needed.        
        get_inference_data -> collects the operations that needs to be carried
        calculate_apply_dw -> self-explanatory. 
        out each time the trainer needs to read/preprocess the inference results
        post -> postprocessing function, merely a placeholder in case it is needed
    The run method iteratively performs get_inference_data, calculate_apply_dw,
    and post.
    """
    
    def __init__(self,net,*args,**kwargs):
        self.net = net
    def init(self):
        pass
    def run(self,*args,**kwargs):        
        self.calculate_apply_dw(self.get_inference_data(),*args, **kwargs)
        self.post(*args,**kwargs)
    def calculate_apply_dw(self, inference_data, *args, **kwargs):
        pass
    def get_inference_data(self,*args,**kwargs):
        pass
    def post(self, *args, **kwargs):
        pass    
    
class SurrGradTrainer(trainer):
    """
    Functional implementation of a trainer class that applies the surrogate
    gradient method to use PyTorch routines to determine the synaptic weight
    adaptations after each network inference.
    It replaces the neuron model's activator with one having a surrogate-
    gradient backward function, and it properly interfaces the net object with
    the PyTorch optimizator language (i.e. finds and pairs the synaptic weights
    as the optimizator parameters)
    """
    def __init__(self,net,trainable_synapse = None,
                 inference_fn = LogSoftmax(dim=1), 
                 loss_fn = NLLLoss(),
                 optimizer = None,**kwargs):
        """
        Generator method, it requires the user to specify the desired inference
        post-process function, the loss function, the targeted synaptic weights
        ("parameters") and the PyTorch optimizer.
        This class has been designed to reproduce the functionality of
        https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
        Assumption: the network has liN output.

        Parameters
        ----------
        net : SHIP network
            network that the user needs training for.
            Assumption: the network has liN output <-- to be modified in future versions
        trainable_synapse : list, optional
            If None, all synaptic weights within the network are considered for
            training. if a list of group tags is instead provided, the algorithm
            restricts training to just the user-provided ones. The default is None.
        inference_fn : function, optional
            Inference results postprocessing function. The default is LogSoftmax(dim=1).
        loss_fn : function, optional
            loss function. The default is NLLLoss().
        optimizer : torch.optim, optional
            PyTorch optimizer. If None is provided, Adam is assumed. The default is None.
        **kwargs : any
            Additional keyword arguments passed to the optimizer.

        Returns
        -------
        None.

        """
        
        super().__init__(net) # at the moment, it merely performs self.net = net
        
        # get parameters
        self.params = []
        if trainable_synapse is None: #all synapse weights are assumed to be parameters
            trainable_synapse = [g.tag for g in self.net.stack if g.is_synapse]
        if not isinstance(trainable_synapse,list):
            trainable_synapse = [trainable_synapse]
        self.params_groups = [g for g in self.net.stack if g.tag in trainable_synapse]
        for gg_label in trainable_synapse:
            if hasattr(self.net.groups[gg_label],'w'):
                self.net.groups[gg_label].w.requires_grad=True
                self.params.append(self.net.groups[gg_label].w)
            if hasattr(self.net.groups[gg_label],'w1'):
                self.net.groups[gg_label].w1.requires_grad=True
                self.params.append(self.net.groups[gg_label].w1)
            if hasattr(self.net.groups[gg_label],'w2'):
                self.net.groups[gg_label].w2.requires_grad=True
                self.params.append(self.net.groups[gg_label].w2)
                
        if not self.params:
            raise Exception('The parameter statement has not been successful (no params found) - please check the provided data')
        
        # optimiser
        if optimizer is None:
            self.optimizer = Adam(self.params, lr=2e-3, betas=(0.9,0.999))
        else:
            self.optimizer = optimizer(self.params,**kwargs)
        # output handler and loss function
        self.inference_fn = inference_fn
        self.loss_fn = NLLLoss()
        
        # overriding the activator functions of the network neurons                    
        for g in self.params_groups:
            for gg in g.source.values():               
                if hasattr(gg, "activator"):
                    if callable(gg.activator):
                        gg.activator = SurrGradSpike.apply
            for gg in g.target.values():
                if hasattr(gg, "activator"):
                    if callable(gg.activator):
                        gg.activator = SurrGradSpike.apply
    
    def init(self):
        """
        Here resetting the history variable loss_hist

        Returns
        -------
        None.

        """
        self.loss_hist = []
        
    def get_inference_data(self):                
        """
        Determines the inference as the network output, here assumed to be 
        a membrane potential [can not find the source of this atm - but this is 
        legit, see literature - TODO - find reference]

        Returns
        -------
        output : tensor
            network winner-take-all-like output.

        """
        # this version assumes a non-firing output
        output,_ = self.net.output.max(dim = 1)        
        return output
        
    def calculate_apply_dw(self,inference_data,labels):
        """
        Here we perform training via PyTorch routines, from the gathered
        inference results

        Parameters
        ----------
        inference_data : tensor
            network inference results.
        labels : tensor
            data targets.

        Returns
        -------
        None.

        """
        
        # calculate loss value
        loss_val = self.loss_fn(self.inference_fn(inference_data), labels)
        # calculate dw
        self.optimizer.zero_grad()
        loss_val.backward()
        # apply dw
        self.optimizer.step()
        # updates loss_hist variable
        self.loss_hist.append(loss_val.item())
        
        
##############################################################################
### activator classes and functions

class SurrGradSpike(Function):
    
    """    
    For more info see:
    Zenke, F. and Ganguli, S. (2018)
    SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks
    Neural Computation 30, 1514–1541 (2018)
    doi:10.1162/neco_a_01086
    https://ganguli-gang.stanford.edu/pdf/17.superspike.pdf
    
    Neftci, E.O., Mostafa, H., and Zenke, F. (2019). 
    Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power 
    of Gradient-based optimization to spiking neural networks. 
    IEEE Signal Processing Magazine 36, 51–63.
    doi: 10.1109/MSP.2019.2931595
    https://ieeexplore.ieee.org/document/8891809
    """
    
    scale = 100.0 # controls steepness of surrogate gradient <-- to be included in some other form

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*t_abs(input)+1.0)**2
        return grad
 
def LIF_autograd_neuron_activator(self,arg,sf = SurrGradSpike.apply):
    return sf(arg)
