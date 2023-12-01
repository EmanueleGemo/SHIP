#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:06:13 2023

@author: egemo
"""

from .component_utils import (refractory,
                              refractory_variabletimestep,
                              delayed,
                              WinnerTakeAll,
                              WinnerTakeAllRefractory)
from .component_models import (inputN,
                               list_inputN,
                               poissonN,
                               lifN,
                               lifN_b,
                               lifN_c,
                               liN,
                               lS_1o,
                               lS_2o,
                               lS_FZenke,
                               wireS,
                               linkS) 
from .plastic_core import (trainer,
                           SurrGradTrainer)
from .data_utils import (TorchVisionDataLoader,
                         TensorHauler,
                         ListHauler,
                         ListHaulerMS,
                         TTFSencoder,
                         Rencoder)
from .core import core
from .network import network
from .utils import Dict,uniform_dist
from .group import group,neurongroup,synapsegroup

__version__ = "0.02"

__all__ = [### essentials
           "core",
           "network",
           
           ### group classes #XXX might remove these ones form all?
           "group",
           "neurongroup",
           "synapsegroup",
           
           ### datatype or utils
           "Dict",
           "uniform_dist",
           
           
           ### data_utils
           "TorchVisionDataLoader",
           "TensorHauler",
           "ListHauler",
           "ListHaulerMS",
           "TTFSencoder",
           "Rencoder",
           # "ISIencoder", #TODO
           
           
           
           ### model wrappers adding functionalities
           "refractory",
           "refractory_variabletimestep", #<- #TODO: to be merged
           "delayed",
           "WinnerTakeAll",
           "WinnerTakeAllRefractory",
           
           ### training 
           "trainer",
           "SurrGradTrainer",
           
           ### models
           "inputN", 
           "list_inputN",
           "poissonN",
           "lifN",
           "lifN_b",
           "lifN_c",
           "liN",
           "lS_1o",
           "lS_2o",
           "lS_FZenke",
           "linkS",
           "wireS"]