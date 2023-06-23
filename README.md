# SHIP
_Spiking (neural network) Hardware In PyTorch_ is an emulation platform intended for hardware-based SNNs, based on compact models to mimic the behavior and functionalitites of the SNN components, and reliant on a PyTorch backend, so to exploit its available conventional machine-learning techniques.

## Platform description 
The models are constructed as a set of time-discrete equations, each applicable to a _group_, i.e. set of hierarchically-identical components of the SNN (neurons and synapses; but also, optionally, learning rule circuit blocks, dendritic integrators, etc.). An easy-to-amend model scaffold is provided, to facilitate the building step of bespoke models.

The temporal progress follows a clock-driven approach, using either a fixed time-step or allowing the user to program time-step changes along the inference.

The group models are all called sequentially during each time-step. The sequence is determined by an ad-hoc routine that minimizes computational artefacts (which are likely for complex, recurrently-connected networks). The data flow across the groups is managed in a supervised approach, though an initialization function (called before the first inference) does most of the job and minimizes any further overhead. 

We note that recurrent networks can be defined in SHIP with the same models used for feedforward network, with no added complexity.

## Interface 
The interface is minimal, yet naturally readable. Network building starts with the instation of an "empty" network. The user can then add the group models, eventually providing both the configurational parameters for each group, and the group's sources and targets. The latter defines the network structure.

The user would then initialize the network, so to consolidate the network object and its inner clockwork.
Eventually, the user can perform inference, optionally providing input data.

Also, the user can set-up a trainer object, which takes care of interfacing the network object in SHIP with the PyTorch training algorithms.

A few utilities are also provided to facilitate download, conversion and handling of datasets.

## Citation
A manuscript preparation is underway, by the provisional title of "Challenges and solutions for the emulation of silicon spiking neural network systems" from Emanuele Gemo, Sabina Spiga, and Stefano Brivio. The target journal is Frontiers In Neuroscience.

## Funding
This work has been funded by the MeM-Scales research project (https://memscales.eu, Horizon2020 grant agreement no. 871371)
