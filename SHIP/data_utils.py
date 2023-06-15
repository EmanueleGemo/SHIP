# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:49:46 2023

@author: Em
"""

from .utils import get_device
from .core import core
from torch import (is_tensor,
                   tensor,
                   arange,
                   manual_seed,
                   randperm,
                   zeros,
                   ones,
                   inf,
                   zeros_like,
                   sparse_coo_tensor,
                   cat,
                   where,
                   logical_and,
                   rand)
from os.path import expanduser
from os import getcwd



###############################################################################
### here we collect helpful stuff for data handling

def TensorHauler(x, y, batch_size = [], max_batches = int(1e9), 
                 shuffle=True, device = None, 
                 data_fun = [], **kwargs):
    """
    Simple data generator, takes tensor data and parses it to be used 
    within an iterator for repeated inference (training) operations

    Parameters
    ----------
    x : tensor
        data to be delivered to the network.
    y : tensor (or vectorial data)
        labels of x.
    batch_size : int, optional
        if provided, it determines the batch_size. The default is [].
    max_batches : int, optional
        Maximum batch number to be yielded. The default is int(1e9).
    shuffle : bool, optional
        If true, the algorithm shuffles the data order. The default is False.
    device : str, optional
        allows to direct the data towards "cpu" or "cuda" devices. 
        The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Yields
    ------
    X : tensor
        x data batch.
    Y : tensor
        y (labels) batch.

    """

    if batch_size:
        core.batch_size = batch_size
        
    if device is None:
        device = get_device()
        
    if not is_tensor(y): #labels datatype check
        y = tensor(y, device = device).long()
        
    number_of_samples = len(x)
    number_of_batches = number_of_samples//core.batch_size ### <- no assertion on batch size
    
    if number_of_batches>max_batches:
        number_of_batches = max_batches
        
    if shuffle:
        if isinstance(shuffle, int):
            manual_seed(shuffle)
        sample_index = randperm(number_of_samples, device = device)    
    else:
        sample_index = arange(number_of_samples, device = device)

    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[core.batch_size*counter:core.batch_size*(counter+1)]
        
        if data_fun:
            X = data_fun(x[batch_index,:],**kwargs)
        else:
            X = x[batch_index,:]
        Y = y[batch_index]

        yield X.to(device=device),Y.to(device=device)

        counter += 1


def ListHauler(x_list, y, batch_size = [], nts = [], max_batches = int(1e9), 
               order="sort", convert_to_tensor = False, device = None, **kwargs):
    """
    This data generator parses list data within an iterator for repeated 
    inference (training) operations, and changes the nts according to the lowest
    value present in each batch (or according to the user-provided min value)

    Parameters
    ----------
    x_list : list
        data to be delivered to the network.
    y : tensor (or vectorial data)
        labels of x.
    batch_size : int, optional
        IF provided, it determines the batch_size. The default is [].
    nts : int, optional
        IF provided, it forces the output duration (number_of_time-steps). 
        The default is [].
    max_batches : int, optional
        Maximum batch number to be yielded. The default is int(1e9).
    order : str, optional
        according to this, the sorting order of the data is decided. 
        options: "shuffle" - shuffles the data, "sort" - sorts the data by
        duration; any other does not touch the sorting order of the original 
        dataset. The default is "sort".
    convert_to_tensor : bool, optional
        if True, the algorithm yields a pytorch tensor in place of a list, 
        formatted according to the provided nts value or to the longest nts
        of the inputs in each batch. The default is False.
    device : str, optional
        delivers the yielded batches to "cpu" or "cuda". If None, the algorithm
        determines automatically which device to use. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Yields
    ------
    X : list
        x data batch.
    Y : tensor
        y (labels) batch.
        
    """
    
    if batch_size:
        core.batch_size = batch_size
        
    units = x_list[0].shape[-1]
    local_nts = nts
        
    if device is None:
        device = get_device()
        
    if not is_tensor(y): #labels datatype check
        y = tensor(y, device = device).long()
        
    number_of_samples = len(x_list)
    len_of_samples = tensor([len(_) for _ in x_list])
    number_of_batches = number_of_samples//core.batch_size
    if number_of_batches>max_batches:
        number_of_batches = max_batches
    
    if order == "shuffle":
        idx = randperm(number_of_samples)
    elif order == "sort":
        idx = len_of_samples.argsort()
    else:
        idx = arange(number_of_samples)

    counter = 0
    
    while counter<number_of_batches:        
        batch_indices = arange(batch_size*counter,batch_size*(counter+1))
        
        if convert_to_tensor:
            X = zeros(core.batch_size,local_nts,units)
            if not nts: # adaptive assumed
                local_nts = len_of_samples[idx[batch_indices]].max()                
            for ii in idx[batch_indices]:
                if len_of_samples[ii]>local_nts:
                    X[ii,:,:] = x_list[ii][:local_nts,:]     
                else:
                    X[ii,:len_of_samples[ii],:] = x_list[ii][:,:]
        else:
            X = [x_list[_] for _ in idx[batch_indices]]
            
        yield X.to(device=device), y[idx[batch_indices]].to(device=device)
        counter += 1
        
    other = number_of_samples%batch_size
    if other and number_of_batches<max_batches:
        tmp = core.batch_size
        core.batch_size = other
        batch_indices = arange(batch_size*counter,batch_size*(counter)+other)
        
        if convert_to_tensor:
            X = zeros(core.batch_size,local_nts,units)
            if not nts: # adaptive assumed
                local_nts = len_of_samples[idx[batch_indices]].max()                
            for ii in idx[batch_indices]:
                if len_of_samples[ii]>local_nts:
                    X[ii,:,:] = x_list[ii][:local_nts,:]     
                else:
                    X[ii,:len_of_samples[ii],:] = x_list[ii][:,:]
        else:
            X = [x_list[_] for _ in idx[batch_indices]]
            
        yield X.to(device=device), y[idx[batch_indices]].to(device=device) # lists may not be loaded to gpu - need to check
        core.batch_size = tmp
        
def ListHaulerMS(x_list, y, params, batch_size = [], nts = [], 
                 max_batches = int(1e9), order="sort", 
                 convert_to_tensor = False, yield_last_param = True,
                 device = None, **kwargs):
    """
    Modification of the ListHauler, this data generator parses list data within 
    an iterator for repeated inference (training) operations AND delivers also
    the time-parameters (list of (dt,nts) tuples) that the run_multistep method
    of the network class can use.

    Parameters
    ----------
    x_list : list
        data to be delivered to the network.
    y : tensor (or vectorial data)
        labels of x.
    params : list
        list of (dt,nts) tuples, of same size of y and x_list, that are to be 
        yielded along with the corresponding data batch
    batch_size : int, optional
        IF provided, it determines the batch_size. The default is [].
    nts : int, optional
        IF provided, it forces the output duration (number_of_time-steps). 
        The default is [].
    max_batches : int, optional
        Maximum batch number to be yielded. The default is int(1e9).
    order : str, optional
        according to this, the sorting order of the data is decided. 
        options: "shuffle" - shuffles the data, "sort" - sorts the data by
        duration; any other does not touch the sorting order of the original 
        dataset. The default is "sort".
    convert_to_tensor : bool, optional
        if True, the algorithm yields a pytorch tensor in place of a list, 
        formatted according to the provided nts value or to the longest nts
        of the inputs in each batch. The default is False.
    yield_last_param : bool, optional
        if True, the algorithm yields only the 1st of the P batch (params 
        subset). The default is True.
    device : str, optional
        delivers the yielded batches to "cpu" or "cuda". If None, the algorithm
        determines automatically which device to use. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Yields
    ------
    X : list
        x data batch.
    Y : tensor
        y (labels) batch.
    P : list
        list of (dt, nts) for the batch X
    
    """
    
    if batch_size:
        core.batch_size = batch_size
        
    units = x_list[0].shape[-1]
    local_nts = nts
        
    if device is None:
        device = get_device()
        
    if not is_tensor(y): #labels datatype check
        y = tensor(y, device = device).long()
        
    number_of_samples = len(x_list)
    len_of_samples = tensor([len(_) for _ in x_list])
    number_of_batches = number_of_samples//core.batch_size
    if number_of_batches>max_batches:
        number_of_batches = max_batches
    
    if order == "shuffle":
        idx = randperm(number_of_samples)
    elif order == "sort":
        idx = len_of_samples.argsort()
    else:
        idx = arange(number_of_samples)

    counter = 0
    
    if yield_last_param:
        P = lambda idx_bb: params[idx_bb[-1]]
    else:
        P = lambda idx_bb: [params[_] for _ in idx_bb]
    
    while counter<number_of_batches:        
        batch_indices = arange(batch_size*counter,batch_size*(counter+1))
        
        if convert_to_tensor:
            X = zeros(core.batch_size,local_nts,units)
            if not nts: # adaptive assumed
                local_nts = len_of_samples[idx[batch_indices]].max()                
            for ii in idx[batch_indices]:
                if len_of_samples[ii]>local_nts:
                    X[ii,:,:] = x_list[ii][:local_nts,:]     
                else:
                    X[ii,:len_of_samples[ii],:] = x_list[ii][:,:]
        else:
            X = [x_list[_] for _ in idx[batch_indices]]
            
        yield X.to(device=device), y[idx[batch_indices]].to(device=device), P[idx[batch_indices]]
        counter += 1
        
    other = number_of_samples%batch_size
    if other and number_of_batches<max_batches:
        tmp = core.batch_size
        core.batch_size = other
        batch_indices = arange(batch_size*counter,batch_size*(counter)+other)
        
        if convert_to_tensor:
            X = zeros(core.batch_size,local_nts,units)
            if not nts: # adaptive assumed
                local_nts = len_of_samples[idx[batch_indices]].max()                
            for ii in idx[batch_indices]:
                if len_of_samples[ii]>local_nts:
                    X[ii,:,:] = x_list[ii][:local_nts,:]     
                else:
                    X[ii,:len_of_samples[ii],:] = x_list[ii][:,:]
        else:
            X = [x_list[_] for _ in idx[batch_indices]]
            
        yield X.to(device=device), y[idx[batch_indices]].to(device=device), P[idx[batch_indices]]
        core.batch_size = tmp   


def TorchVisionDataLoader(dset:str = "MNIST", 
                          standardize:bool = True, 
                          root = [], 
                          labels = [], 
                          **kwargs):
    """
    Helper to rapidly access to some of the TorchVision dataset. 
    TorchVision dependency. Applicable for the following (see notes):
        CIFAR10 [C], 
        EMNIST [E] (need to specify the split as "byclass", 
                    "bymerge", "balanced", "letters", "digits" 
                    or "mnist" - see help), 
        FashionMNIST [F],
        KMNIST [K], 
        MNIST [M],
        Places365 [P] (untested - see help),
        QMNIST [Q] (untested - see help),
        USPS [U]
    

    Parameters
    ----------
    dset : str
    dataset to be loaded. Default: "MNIST"
    standardize: bool
        If True, the algorithm reshapes the data onto a set of unidimensional 
        samples, andit converts its values from 0-peak (int?) to 0-1 (float). 
        The default is True.
    root : str, optional
        Path to the MNIST folder.
        The default is expanduser(getcwd()+'\\data\\datasets\\torch\\'+dset).
    labels : tensor, optional
        It allows to select samples based on the labels. The default is [].

    Returns
    -------
    xtrain : tensor
        vector containing the training dataset.
    ytrain : tensor
        vector containing the training dataset labels.
    xtest : tensor
        vector containing the test dataset..
    ytest : tensor
        vector containing the test dataset labels.
    """
    
    if dset == "C" or dset == "c":
        dset = "CIFAR10"
    elif dset == "E" or dset == "e":
        dset = "EMNIST"
    elif dset == "F" or dset == "f":
        dset = "FashionMNIST"
    elif dset == "K" or dset == "k":
        dset = "KMNIST"
    elif dset == "M" or dset == "m":
        dset = "MNIST"
    elif dset == "P" or dset == "p":
        dset = "Places365"
    elif dset == "Q" or dset == "q":
        dset = "QMNIST"
    elif dset == "U" or dset == "u":
        dset = "USPS"
    
    if not root:
        root = expanduser(getcwd()+'\\data\\datasets\\torch\\'+dset)
        
    if "torchvision" not in dir():
        import torchvision
        
    if dset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root, train=True, 
                                                     transform=None, 
                                                     target_transform=None, 
                                                     download=True,**kwargs)
        test_dataset = torchvision.datasets.CIFAR10(root, train=False, 
                                                    transform=None, 
                                                    target_transform=None, 
                                                    download=True,**kwargs)
    elif dset == "EMNIST":
        train_dataset = torchvision.datasets.EMNIST(root, train=True, 
                                                    transform=None, 
                                                    target_transform=None, 
                                                    download=True,**kwargs)
        test_dataset = torchvision.datasets.EMNIST(root, train=False, 
                                                   transform=None, 
                                                   target_transform=None, 
                                                   download=True,**kwargs)
    elif dset == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(root, train=True, 
                                                          transform=None, 
                                                          target_transform=None, 
                                                          download=True,**kwargs)
        test_dataset = torchvision.datasets.FashionMNIST(root, train=False, 
                                                         transform=None, 
                                                         target_transform=None, 
                                                         download=True,**kwargs)
    elif dset == "KMNIST":
        train_dataset = torchvision.datasets.KMNIST(root, train=True, 
                                                    transform=None, 
                                                    target_transform=None, 
                                                    download=True,**kwargs)
        test_dataset = torchvision.datasets.KMNIST(root, train=False, 
                                                   transform=None, 
                                                   target_transform=None, 
                                                   download=True,**kwargs)
    elif dset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(root, train=True, 
                                                   transform=None, 
                                                   target_transform=None, 
                                                   download=True,**kwargs)
        test_dataset = torchvision.datasets.MNIST(root, train=False, 
                                                  transform=None, 
                                                  target_transform=None, 
                                                  download=True,**kwargs)
    elif dset == "Places365":
        train_dataset = torchvision.datasets.Places365(root, train=True, 
                                                       transform=None, 
                                                       target_transform=None, 
                                                       download=True,**kwargs)
        test_dataset = torchvision.datasets.Places365(root, train=False, 
                                                      transform=None, 
                                                      target_transform=None, 
                                                      download=True,**kwargs)
    elif dset == "QMNIST":
        train_dataset = torchvision.datasets.QMNIST(root, train=True, 
                                                    transform=None, 
                                                    target_transform=None, 
                                                    download=True,**kwargs)
        test_dataset = torchvision.datasets.QMNIST(root, train=False, 
                                                   transform=None, 
                                                   target_transform=None, 
                                                   download=True,**kwargs)
    elif dset == "USPS":
        train_dataset = torchvision.datasets.USPS(root, train=True, 
                                                  transform=None, 
                                                  target_transform=None, 
                                                  download=True,**kwargs)
        test_dataset = torchvision.datasets.USPS(root, train=False, 
                                                 transform=None, 
                                                 target_transform=None, 
                                                 download=True,**kwargs)

    xtrain = train_dataset.data
    ytrain = train_dataset.targets
    xtest = test_dataset.data
    ytest = test_dataset.targets

    if standardize:
        peak = xtrain.max().float()
        xtrain = xtrain.float().div(peak).reshape(xtrain.shape[0],-1)
        xtest = xtest.float().div(peak).reshape(xtest.shape[0],-1)
    
    if labels:
        mask_test = zeros((len(labels),xtest.shape[0]),dtype = bool)
        for ii in range(len(labels)):
            mask_test[ii,:] = ytest == labels[ii]
        mask_test = mask_test.any(dim=0)
        mask_train = zeros((len(labels),xtrain.shape[0]),dtype = bool)
        for ii in range(len(labels)):
            mask_train[ii,:] = ytrain == labels[ii]
        mask_train = mask_train.any(dim=0)
    else:
        mask_test = ones(len(xtest),dtype = bool)
        mask_train = ones(len(xtrain),dtype = bool)
    return xtrain[mask_train,:], ytrain[mask_train], xtest[mask_test,:], ytest[mask_test]

def logI(x, tau = 0.02, thr = 0, tmax = inf):
    """
    Utility converting intensity data according to a logarithmic function 
    out = -tau*log(x).

    Parameters
    ----------
    x : tensor
        data to be converted, assumed to be comprised between 0 - (clamped at) 1
    tau : float, optional
        Temporal constant driving the log function. The default is 0.02.
    thr : float, optional
        Intensity threshold below which the output is capped. The default is 0.2.
    tmax : float
        Output value above threshold. The default is infinity.

    Returns
    -------
    out : tensor
        Transformation of the data in x. Can be used as time2firstspike converter.
        
    """
    out = -tau*(x.clamp(max=1)).log()
    out[x<=thr] = tmax # check for data below threshold    
    return out

def linI(x, min_rate = 0, max_rate = 100, thr=0.2):
    """
    Utility converting intensity data according to a linear function 
    out = tau*(x-thr).

    Parameters
    ----------
    x : tensor
        data to be converted, assumed to be comprised between 0 - (clamped at) 1
    min_rate : float, optional
        value assigned at 0. The default is 0.
    max_rate : float, optional
        value assigned at 1. The default is 100.
    thr : float, optional
        Intensity threshold, below which data is assingned to min_rate. 
        The default is 0.2.

    Returns
    -------
    out : tensor
        Transformation of the data in x. Can be used as time2spikerate converter.
        
    """
    out = min_rate+(max_rate-min_rate)*x.clamp(max = 1)
    out[x<=thr] = min_rate
    return out

def TTFSencoding(x,nts,dt,preprocess = logI,jitter = zeros,seed:int = [],**kwargs):
    """
    This function converts an x tensor (float, 0..1 values) of absolute 
    intensity values, of size [samples-units], to a dense tensor of size 
    [samples-nts-units], according to a simple time-to-first-spike method.
    The intensity data is pre-processed with the function 
    preprocess (logI by default, **kwargs will be delivered to it). 
    The argument jitter is an additional function that can be used to inject 
    temporal noise [s] to the method output.

    Parameters
    ----------
    x : tensor
        data to be converted (float), assumed of size[samples,units]
    nts : int
        number of time-steps.
    dt : float
        time-step size.
    preprocess : function, optional
        Function detemining the data-to-time2firstspike conversion. 
        The default is logI.
    jitter : function, optional
        Function determining the temporal fluctiation of each spike. 
        The default is zeros (no fluctuation)
    seed : int, optional
        if provided, it is used in manual_seed before the conversion.
        The default is [].
    **kwargs : any
        keyword arguments to be sent to preprocess.

    Returns
    -------
    out : tensor
        boolean tensor of the converted data, of size [samples, nts, units].

    """
    if seed:
        manual_seed(seed)
    TTFS = ((preprocess(x,**kwargs)+jitter(x.shape))/dt).long().clamp(min=-1,max=nts)
    mask = logical_and(TTFS>=0,TTFS<nts)
    b,u = where(mask)
    t = TTFS[mask]
    out = zeros(x.shape[0],nts,x.shape[1])
    out[b,t,u] = True
    return out[:,:-1,:]
  
def Rencoding(x,nts,dt,preprocess = linI,seed: int = [],**kwargs):
    """
    Fast utility that converts an x tensor (float, 0..1 values) of absolute 
    intensity values, of size [samples-units], to a dense tensor of size 
    [samples-nts-units] containing the spiking rate-converted data (poisson spiking)
    The spiking rate conversion is dictated by the argument method (linear by 
    default, **kwargs will be delivered to it).

    Parameters
    ----------
    x : tensor
        data to be converted (float), assumed of size[samples,units]
    nts : int
        number of time-steps.
    dt : float
        time-step size.
    preprocess : function, optional
        preprocess function, detemining the spiking-rate conversion. 
        The default is linI.
    seed : int, optional
        if provided, it is used in manual_seed before the conversion.
        The default is [].
    **kwargs : any
        keyword arguments to be sent to preprocess.

    Returns
    -------
    out : tensor
        boolean tensor of the converted data, of size [samples, nts, units].

    """
    if seed:
        manual_seed(seed)
    R = preprocess(x,**kwargs) #rate [Hz]
    out = rand(x.shape[0],nts,x.shape[1])<(R*dt).unsqueeze(1)
    return out

# def ISIencoding(x,nts,dt, preprocess = linI, jitter = zeros, 
#                 randomize_first:bool = True, burst_spike_cap:int = [], 
#                 seed: int = [],**kwargs): <-- TODO - complete
#     """
#     Similarly to Rencoding, this converts an x tensor (float, 0..1 values) of  
#     absolute intensity values, of size [samples-units], to a dense tensor of 
#     size [samples-nts-units] containing the inter-spike-interval converted data.
#     The spiking rate conversion is dictated by the argument method (linear by 
#     default, **kwargs will be delivered to it).
#     jitter can be added to each spike according to the jitter function argument
#     (default: zeros)

#     Parameters
#     ----------
#     x : tensor
#         data to be converted (float), assumed of size[samples,units]
#     nts : int
#         number of time-steps.
#     dt : float
#         time-step size.
#     preprocess : function, optional
#         Function detemining the spiking-rate conversion. 
#         The default is linI.
#     jitter : function, optional
#         Function determining the temporal fluctiation of each spike. 
#         The default is zeros (no fluctuation)
#     randomize_first : bool, optional
#         If True, it shifts the temporal axis within the allowed period.
#     burst_spike_cap : int, optional
#         If provided, the algorithm caps the number of spikes up to the maximum
#         burst_spike_cap, thus allowing to perform burst-encoding with the same 
#         algorithm. CAREFUL - no assertion on nts, might not be sufficient.
#     seed : int, optional
#         if provided, it is used in manual_seed before the conversion.
#         The default is [].
#     **kwargs : any
#         keyword arguments to be sent to preprocess.

#     Returns
#     -------
#     out : tensor
#         boolean tensor of the converted data, of size [samples, nts, units].

#     """
#     if seed:
#         manual_seed(seed)
#     R = preprocess(x,**kwargs) #rate [Hz]
#     if randomize_first:
#         rf = rand(R.shape)
#     else:
#         rf = 0
#     prob = R.unsqueeze(1).expand(R.shape[0],nts,R.shape[1])*arange(nts)*(dt*R.unsqueeze(1).expand(nts)).cumsum(dim=1)
#     if randomize_first:
#         prob = prob+.unsqueeze(1).expand(nts)
    
#     .fmod(1)==0
    
    
#     # out = rand(x.shape[0],nts,x.shape[1])<(R*dt).unsqueeze(1)
#     # 
#     return out