from SHIP import (network,
                  liN,                  
                  inputN,
                  SurrGradTrainer as trainer)
from braille_task_utils import (load_braille_dataset,
                                delta_encoding,
                                izhikevicN,
                                ilS_2o,
                                select_datasets,
                                train1epoch,
                                recurrent_topology)
from os import getcwd
from tqdm import tqdm
from torch import manual_seed,rand,cat,randperm,zeros,arange
from torch.optim import Adam

### main parameters
dt = 0.0005
nts = 300
total_time = dt*nts
encoding_params = {"nts":nts, # parameters used for the spike encoding
                   "bits":1,
                   "delta":0.01,
                   "interp":10,
                   "swap":True}
test_train_ratio = .2

##### get and convert dataset (delta-rule) ####################################
# note: the dataset has been previously downloaded
# (https://www.kaggle.com/datasets/shanks0465/braille-character-dataset)
# and then saved using the pickle module for the sake of convenience
folder = getcwd()+"//"
filename = "braille_dset.pic" 
x,y = load_braille_dataset(folder,filename)
xe = delta_encoding(x,**encoding_params)
xT,yT,xt,yt = select_datasets(xe,y,test_train_ratio = test_train_ratio,
                              limitclasses = 10, seed = 150)
   
##### network building stage
manual_seed(200)
classes = 1+yT.max()
datashape = xT.shape[2]  
n_model = izhikevicN
s_model = ilS_2o
Nrec = 125
Nhid = 128
# generating the input-to-recurrent weight map

imap = rand(Nrec,)<0.25
num_connections = 4
locs = cat([randperm(Nrec)[~imap][:datashape].unsqueeze(-1) for _ in range(num_connections)],dim=1)
w = zeros(datashape,Nrec)
w[cat([arange(datashape).unsqueeze(-1) for _ in range(num_connections)],dim=1).reshape(-1),locs.reshape(-1)] = 1

# generating the recurrent-to-recurrent weight map
wr = recurrent_topology([5,5,5],imap)

# building the network
net = network()
net.add(inputN,'I',N = datashape)
net.add(n_model,'L0',N = Nrec)#1024
net.add(n_model,'L1',N = Nhid)
net.add(liN,'out',N = classes, is_output = True)
net.add(s_model,'S0',source='I',target = 'L0',w_scale = 1000,w = w)
net.add(s_model,'S1',source='L0',target = 'L1',w_scale = 800)
net.add(s_model,'S2',source='L0',target = 'L0',w_scale = 1500,w = wr)
net.add(s_model,'S3',source='L1',target = 'out',w_scale = 100)

###### test ###################################################################
# net.set_monitor(**{'L0':['u','v'],'L1':['u','v'],'L2':['u','v'],'out':['u']})
# net.set_monitor(**{'L0':['u','v'],'L1':['u','v'],'out':['u'],'S2':['output']})
# net.init(dt = dt, nts = nts, batch_size = 1)
# net.run(xT[[0],:,:])
# data = net.get_monitored_results()

##### network training ########################################################
batch_size = 16
net.init(dt = dt, nts = nts, batch_size = batch_size)
epochs = 300
save_cnt = 5

# building trainer
trn = trainer(net,optimizer = Adam,lr = 10e-3,weight_decay = 0)
trn.init()

# now training!
training_ac,test_ac,loss_hist = None,None,None    
mod = epochs%save_cnt
noise = None
cm = None
if mod>0:
    epochs = epochs + save_cnt - mod     
with tqdm(range(epochs)) as pbar:
    for ee in pbar:
        (net,trn,
         training_ac,test_ac,
         loss_hist,
         local_cm,
         status) = train1epoch(net,
                               trn,
                               xT,yT,
                               xt,yt,
                               ee,                  
                               training_ac,
                               test_ac,
                               loss_hist,
                               classes,
                               save_cnt = save_cnt)
        if status:
                pbar.set_postfix_str(status)
print('training done')
