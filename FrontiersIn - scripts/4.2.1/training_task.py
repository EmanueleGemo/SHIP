"""
This script serves as a guideline for the reproduction of the data shown in 4.2.2.

The data there presented was obtained with a pre-release version of SHIP, which
has then been updated since. This script converts the work done before to a
version that uses the new release of SHIP. However, the hyperparameter
optimization may still be ineffective for this version, due to minor but
important changes to the models.
Specifically, the phiIR and phiRR have been set to 1 and 1 respectively, which
might not be as effective as desired now. Thus, the quantitative results of 
this script may be different from the ones shown in the article.
"""

from torch import (tensor,
                   rand)
                   # einsum,ceil,div,clamp,rand,randperm
from tqdm import tqdm
from SHIP import SurrGradTrainer as trainer
from spoken_digit_utils import (get_spiking_input,
                                build_conversion_net,
                                select_traintestdata,
                                build_trainable_net,
                                train_1_epoch,
                                reservoir_spikes)
from torch.optim import Adam as optimizer
trn_kwargs = {'lr' : 1e-3,
              'betas' : (0.9, 0.999),
              'weight_decay' : 0.01}

##### parameters ##############################################################
dt = 0.001
nts = 300 # just a placeholder
batch_size = 16
# the dataset has been downloaded from the repository and saved as a pickle
# file, which contains the following:
# {'x_list': Nsized_list_of_wav_numpy_arrays,
#  'y_train': Nsized_array_of_class_labels}
# justgetthesamples instead saves the converted cochleograms (due to the Lyon module not working on all platforms)

dataset_fname = "spokendigit.pickle"
backup_dsetfname = "justgetthesamples.pic"
backup_encodeddsetfname = "dset4quant.pic" # for the following quantization task

### lyon model and spike generation parameters
Lyon_params = {'sample_rate':8000,#Hz
               'decimation_factor':64,
               'ear_q':8,
               'step_factor':0.32,
               'differ':True,
               'agc':True,
               'tau_factor':0.5,
               'dt':dt} #s
deltaencoder_params = {'dt':dt,
                       'max_rate':500,#Hz
                       'thr':0.5}

# reservoir net parameters
Nr_dim = [5,5,5]     #[4,4,16]
NR = tensor(Nr_dim).prod().item() # number of reservoir neurons
NI = int(Lyon_params['ear_q']/Lyon_params['step_factor']*2) # input neurons
IEratio = 0.2 # ratio between reservoir's excitatory and inhibitory neurons
C = 4 # each input neuron connects to C randomly chosen reservoir neurons
imap = rand(NR)<IEratio # map of inhibitory reservoir neurons
taE = 8e-3 # excitatory synapses tau_alpha [s]
taI = 4e-3 # inhibitory synapses tau_alpha [s]
delay_timeRR = 1e-3 # reservoir-to-reservoir synapses delay time
thrR = 0.02 # reservoir neurons threshold
phiIR = 1 # scaling factor for the input-to-reservoir synaptic weights
phiRR = 1 # scaling factor for the reservoir-to-reservoir synaptic weights
wI = 64 # max weight value (input-to-reservoir synaptic weights)
tbR = 64e-3 # reservoir tau beta [s]
refr_timeR = 3e-3 #reservoir neurons refractory time [s]

### training net parameters
tbO = 65e-3 # tau_beta output neurons
taRO = taE # tau_alpha reservoir-to-output synapses [s]
NO = 10 # number of output neurons (10 for the spoken digit dataset)
phiRO = 30 # reservoir-to-output synaptic weight scaling factor

### data and training parameters:
epochs = 100
train_percentage = 0.85
noise = 1/100
shuffle = False
save_cnt = 10


##### get spoken digit datasets post-reservoir ################################
spks,labels = get_spiking_input(dataset_fname,Lyon_params,deltaencoder_params,backup_dsetfname)
net = build_conversion_net(NI,NR,
                           thrR,tbR,refr_timeR,
                           taI,taE,
                           phiRR,delay_timeRR,
                           phiIR,
                           Nr_dim,imap,C,
                           seed = 1000, 
                           wI = wI,
                           dt = 0.001, nts = 300, batch_size = batch_size)
x,y = reservoir_spikes(net,spks,labels,batch_size = batch_size)      
(xtest,ytest,
 xtrain,ytrain,
 test_indices) = select_traintestdata(x,y,
                                      train_percentage,
                                      seed = 2000,
                                      dsetfname = backup_encodeddsetfname)


##### train network ###########################################################
training_ac,test_ac,loss_hist = None,None,None
snn = build_trainable_net(NR,
                          NO,
                          tbO,
                          taRO,
                          phiRO,
                          seed = 3000,
                          dt = dt, nts = nts, batch_size = batch_size)

# set up trainer
trn = trainer(snn,trainable_synapse = 'io', optimizer = optimizer, **trn_kwargs)
trn.init()

# training process
print('here training the parameters:')
with tqdm(range(epochs), total = epochs) as pbar:
    for ee in pbar:
        (snn,trn,
         training_ac,
         test_ac,
         loss_hist,
         res,
         status)= train_1_epoch(snn,
                                trn,
                                xtrain,ytrain,
                                xtest,ytest,
                                ee,
                                shuffle,
                                training_ac,
                                test_ac,
                                loss_hist,
                                noise = noise,
                                save_cnt = save_cnt)
        if status:
                pbar.set_postfix_str(status)