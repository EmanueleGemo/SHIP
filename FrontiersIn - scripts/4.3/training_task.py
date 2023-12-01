"""
Script that instates and train a LSM to recognize the 18 classes of the 
MIT-BIH arrhythmya dataset.

To work out the data shown in Fig 12, one needs to unhide the sections
commented away. Likely, due to changes of the platform, the hyperparameters
need tweaking to converge towards the same results as the one shown in Fig 12 
(the data there shown has been obtained with a pre-release version of SHIP)

"""

from ecg_task_utils import (get_datasets,
                            one_run,
                            get_network_behavior)
from torch import zeros
##### all parameters ##########################################################

# global
epochs = 50
dt = .5e-3 # s
batch_size = 16
num_of_iterations = 1

# dataset
train_dataset_maxsize = 500
test_dataset_maxsize = 50
bits = 3 # how many bits are used for the delta encoding algorithm

# network building
tau_beta = 65e-3
refr_time = 2e-3
tau_alpha = 8e-3
Nr_dim = [6,6,6] # also 5,5,5 works fine
IEratio = 0.2 # ratio between inhibitory and excitatory neurons in reservoir

##### get dataset #############################################################
xtrain,ytrain, xtest,ytest, _,_,_,_ = get_datasets(bits = bits,
                                                   dt = dt,
                                                   train_dataset_maxsize = train_dataset_maxsize, 
                                                   test_dataset_maxsize = test_dataset_maxsize)

NI = xtrain[0].shape[-1]

# # here trimming data (unhide for this to work)
# checked_classes = ["R", # Right bundle branch block beat
#                    "A", # Atrial premature beat
#                    "V",	# Premature ventricular contraction
#                    "F"]	# Fusion of ventricular and normal beat
# ll = [2,3,7,8] # (positional of the checked classes)
# m = zeros_like(ytrain,dtype = bool)
# for l in ll:
#     m = logical_or(m,ytrain==l)
# ytrain = ytrain[m]
# xtrain = [_ for i,_ in enumerate(xtrain) if m[i]]
# m = zeros_like(ytest,dtype = bool)
# for l in ll:
#     m = logical_or(m,ytest==l)
# ytest = ytest[m]
# xtest = [_ for i,_ in enumerate(xtest) if m[i]]
# for i,l in enumerate(ll):
#     ytrain[ytrain == l] = i
#     ytest[ytest==l] = i

output_count = len(ytest.unique())


##### train networks and get results ##########################################

reservoir_tau_beta = [10e-3,30e-3,65e-3,.1,.33,1,10,100]
reservoir_threshold = [0.02] # <-fixed
w_reservoir_scaling_factor = [.1,.25,.375,.5,.625,.75,.9]
w_input_scaling_factor = [0.5] #<- fixed

training_results = zeros((num_of_iterations,
                          len(reservoir_tau_beta),
                          len(reservoir_threshold),
                          len(w_reservoir_scaling_factor),
                          len(w_input_scaling_factor)))

# unhide to get monitored results
# avg_spike_count = training_results.clone()
# avg_current = avg_spike_count.clone()

for ii,nn in enumerate(range(num_of_iterations)): 
    for i1,bb in enumerate(reservoir_tau_beta):
        for i2,tt in enumerate(reservoir_threshold):
            for i3,ww in enumerate(w_reservoir_scaling_factor):
                for i4,ii in enumerate(w_input_scaling_factor):
                    snn,training_results[ii,i1,i2,i3,i4] = one_run(dt,
                                                                   batch_size,
                                                                   nn,
                                                                   NI,
                                                                   Nr_dim,
                                                                   bits,
                                                                   IEratio,                  
                                                                   refr_time,
                                                                   tau_alpha,
                                                                   tau_beta,
                                                                   
                                                                   bb,tt,ww,ii,
                                                                   
                                                                   output_count,
                                                                   nn, # seed
                                                                   
                                                                   xtrain,ytrain,
                                                                   xtest,ytest,
                                                                   epochs)
                    
                    # avg_spike_count, avg_current = get_network_behavior(snn)
                
                
                
                    
                    
                
    