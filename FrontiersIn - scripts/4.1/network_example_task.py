"""
This script generates the data shown in Figure 8b.
"""

# import classes and functions
from SHIP import (network, # network class
                  inputN, # input neuron class
                  lS_1o, # 1st order leaky synapse class
                  lifN, # LIF neuron class
                  refractory) # refractory superclass
from torch import (manual_seed,
                   rand,
                   arange,
                   zeros,
                   normal)

# preliminary ops, determining emulated time and other minor details
eps = 1e-6 #small number
batch_size = 10 #number of parallel simulations
time, dt = .1, 1e-4 #emulated time [seconds], time-step size [seconds]
nts = int(time//dt + time%dt) #number of time-steps
ns = [3,1] #network neuron layer size

# input generation
rate = 20 #[Hz]
manual_seed(3000)
poisson_input = (rand(nts,ns[0])<(rate*dt)).expand(batch_size,nts,ns[0])

# define network
snn = network()
# add neuron groups
snn.add(inputN, #add input_neuron group
'I', #group tag (mandatory)
N = ns[0]) #number of units within the group
snn.add(refractory(lifN), 'N1', N = ns[1], #add refractory LIF group
        tau_beta = arange(10e-3,100e-3+eps,10e-3).unsqueeze(-1), #temporal constant [s]        
        thr = 1.,#threhshold potential [a.u]
        _u_ = lambda b,n: normal(zeros(n),1).abs().unsqueeze(0).expand(b,n), #potential at t = 0 [a.u.]        
        _u0_ = 0., #rest/reset potential
        refr_time = 10e-3) #refractory time [seconds]
# add synaptic group
snn.add(lS_1o, 'S1', source = 'I', target = 'N1', #add synapse group
        w_scale = 150, #synaptic weight global scaling factor
        tau_alpha = 5e-3, #temporal constant [s]
        delay_time = 0e-3, #delay time
        w__ = rand) # synaptic weight matrix

# init network and monitors
snn.set_monitor(**{'S1':['output','I'],'N1':['u','output']})
snn.init(dt = dt, nts = nts, batch_size = batch_size)
# run simulation and gather data
snn.run(poisson_input) # run emulation
data = snn.get_monitored_results() # gather data

