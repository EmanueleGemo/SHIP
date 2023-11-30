"""
Here we compare the calculation time for the simulation of a large network
(10^6 parameters) on SHIP, Brian2 and RockPool.
"""

### main features
rate = 100 # Hz
N = 500 # neurons per layer
nts = 1000 # number of time-steps
DT = 0.001 # time-step size
tau_beta = 0.01 # LIF neuron dynamics
tau_alpha = 0.001 # leaky synapse dynamics

################################## SHIP 
# import libraries
from SHIP import (network,lifN,lS_1o,poissonN)
from torch import rand
from time import time

st = time()
# create network and add neuron layers/synapses sequentially
net = network()
layers = ['I','N1','N2','N3','O']
net.add(poissonN,'I',N = N, rate = rate)
for ii,ll in enumerate(layers):
    if ii ==0:
        continue
    else:
        net.add(lifN,ll,N=N,tau_beta = tau_beta)
        net.add(lS_1o,layers[ii-1]+ll,source=layers[ii-1],target=ll,tau_alpha = tau_alpha,w__ = rand)
# init monitor, network and run simulation
net.set_params('O',is_output = True)
net.init(dt = DT,nts = nts,batch_size = 1)
net.run()

print("SHIP calc. time: %0.3f seconds"%(time()-st))


################################## Brian2
# import libraries
from brian2 import (ms,mV,Hz,second,SpikeMonitor,run,PoissonGroup,NeuronGroup,
                    Synapses,defaultclock)

st = time()
# set default quantities
defaultclock.dt = DT*second
taum = tau_beta*second
taue = tau_alpha*second
Vt = 1000*mV
Vr = 0*mV
# set a model for neurons and synapses (just one unit)
eqs = '''
dv/dt  = (ge-v)/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
'''
# create network
P0 = PoissonGroup(N,rate*Hz)
def layer(previous):
    nn = NeuronGroup(N, eqs, threshold='v>Vt', reset='v = Vr', refractory=1*ms, method='exact')
    nn.ge = 0*mV
    ss = Synapses(previous, nn, 'we : volt', on_pre='ge += we')
    ss.connect()
    ss.we = 'j*rand() * volt'
    return nn, ss
N1,S1 = layer(P0)
N2,S2 = layer(N1)
N3,S3 = layer(N2)
N4,S4 = layer(N3)
# init monitoring and run simulation
s_mon = SpikeMonitor(N4)
run(1 * second)

print("Brian2 calc time: %0.3f seconds"%(time()-st))


################################## RockPool
#import libraries
from rockpool.nn.modules import (LIF,Module)
from rockpool.parameters import Parameter
import numpy as np

st = time()
# create network class
class net(Module):
    def __init__(self, shape: tuple, *args, **kwargs):
        super().__init__(shape, *args, **kwargs)
        # - Configure weight parameters
        self.w_IN1 = Parameter(
            shape=self.shape[0:2],
            family="weights",
            init_func=lambda size: np.random.normal(size=size),
        )
        self.w_N1N2 = Parameter(
            shape=self.shape[0:2],
            family="weights",
            init_func=lambda size: np.random.normal(size=size),
        )
        self.w_N2N3 = Parameter(
            shape=self.shape[0:2],
            family="weights",
            init_func=lambda size: np.random.normal(size=size),
        )
        self.w_N3N4 = Parameter(
            shape=self.shape[0:2],
            family="weights",
            init_func=lambda size: np.random.normal(size=size),
        )
        # - Build submodules
        self.N1 = LIF((N),tau_mem=0.01,tau_syn=0.001,spiking_input=True,spiking_output=True)
        self.N2 = LIF((N),tau_mem=0.01,tau_syn=0.001,spiking_input=True,spiking_output=True)
        self.N3 = LIF((N),tau_mem=0.01,tau_syn=0.001,spiking_input=True,spiking_output=True)
        self.N4 = LIF((N),tau_mem=0.01,tau_syn=0.001,spiking_input=True,spiking_output=True)
    def evolve(self, input, record: bool = False) -> (np.array, np.array, dict):
        # - Initialise output arguments
        new_state = {}
        record_dict = {}
        # - Pass input through the input weights and submodule
        x, mod_state, mod_record_dict = self.N1(
            np.dot(input, self.w_IN1), record
        )
        x, mod_state, mod_record_dict = self.N2(
            np.dot(x, self.w_N1N2), record
        )
        x, mod_state, mod_record_dict = self.N3(
            np.dot(x, self.w_N2N3), record
        )
        output, mod_state, mod_record_dict = self.N4(
            np.dot(x, self.w_N3N4), record
        )
        # - Maintain the submodule state and recorded state dictionaries
        new_state.update({"mod_recurrent": mod_state})
        record_dict.update({"mod_recurrent": mod_record_dict})
        # - Return outputs
        return output, new_state, record_dict
# instatiate network
net_mod = net(shape=(N,N,N,N,N))
# get spiking input
time_base = np.arange(nts) * DT
input_clocked = np.random.rand(nts, N)<1
# run simulation
output,state,_ = net_mod(input_clocked)

print("RockPool calc time: %0.3f seconds"%(time()-st))




