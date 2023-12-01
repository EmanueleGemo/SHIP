"""
gathering the accuracy as a function of the weight quantization
ideally, this script would be run the various results obtained with the trainig_task.py script
"""

from SHIP import ListHauler,lS_1o
from torch import (tensor,
                   arange,
                   zeros,
                   normal,
                   manual_seed,
                   eye,
                   no_grad,
                   cat,
                   inf as torch_inf,
                   logical_and)
from spoken_digit_utils import (build_trainable_net)
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# global params
cm = 1/2.54  # centimeters in inches <- plot purposes
dt = 0.0005
nts = 300
batch_size = 16
fname = "results_spoken_digit_86.5.pic" #<-- we arbitrary select a savefile
dsetfname = "dset4quant.pic"

### training net parameters
Nr_dim = [5,5,5]; NR = tensor(Nr_dim).prod().item()
NO = 10 # number of output neurons (10 for the spoken digit dataset)

tbO = 65e-3 # tau_beta output neurons
taRO = 8e-3 # tau_alpha reservoir-to-output synapses [s]
phiRO = 30 # reservoir-to-output synaptic weight scaling factor

def quantize(w,n,l,r,sigma = None,seed = None):        
    if n:
        # generating a lookup table [left boundary, central value, right boundary]
        eps = 1e-5
        delta = abs(l-r)/(n-1)    
        rng = arange(l,r+eps,delta).unsqueeze(-1)
        tb = rng.expand(n,3)
        tb = cat((rng-delta/2,rng,rng+delta/2),dim=1)
        tb[1:,0] = tb[:-1,2].clone()
        tb[0,0] = -torch_inf
        tb[-1,-1] = torch_inf
        
        # using the table to yield quantized values (central ones)
        maps = logical_and(
            w.view(w.numel(),-1)>=tb[:,0], 
            w.view(w.numel(),-1)<tb[:,2])            
        w = tb[:,1].masked_select(maps).view(w.shape)
        
        # applying normal distribution
        if seed:
            manual_seed(seed)
        if sigma is None:
            sigma = (r-l)/n/6 # here assuming a regular, 3 sigma separation
        w = normal(w,sigma)        
    return w

# class lS_1o_q(lS_1o): # <-- alternatively, we can integrate the quantization in the synapse model (not useful here)
#     """
#     integrating the drift functionality, updating the set_initial_state method
#     """    
#     parameters = lS_1o.parameters | {'w0': None, #<-- as-written weight (saved in set_initial_state)
#                                      'n':None, # number of levels
#                                      'l':None, # left boundary
#                                      'r':None, # right boundary
#                                      's':None} # normal distribution sigma (scalar value assumed here)
#     @staticmethod
    
        
#     def set_initial_state(self, *args, quantize = False, seed = None):
#         super().set_initial_state(*args)
#         if self.w0 is None: 
#             self.w0 = self.w #<-initial storage            
#         if quantize:  
#             with no_grad():
#                 self.w = self.qf(self.w0,self.n,self.l,self.r,self.s,seed)
                  

##### main ####################################################################

# retrieve encoded datasets
with open(dsetfname, "wb") as h:
    (xtest,ytest,xtrain,ytrain,test_idx) = pickle.load(h)
    
# retrieve trained synaptic weights and their boundary
with open(fname, "wb") as h:
    data = pickle.load(h)
    w = data['w_io']
    m = w.abs().max()    
    
# build network   
snn = build_trainable_net(NR,
                          NO,
                          tbO,
                          taRO,
                          phiRO,
                          s_model = lS_1o,
                          w = w, # <- placeholder, will be changed in iteration
                          seed = 3000,
                          dt = dt, nts = nts, batch_size = batch_size)

snn.set_params('io',l=-m,r=m)

investigated_levels = tensor([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,48,64])
seeds = arange(10).tolist()

ac_array = zeros(len(investigated_levels),len(seeds))

for zz,ss in enumerate(seeds):
    res_matrix = zeros(len(investigated_levels),NO,NO)
    for kk,ll in tqdm(enumerate(investigated_levels)):    
        
        # setting network and local results for new run
        snn.set_params('io',w = quantize(w,-m,m,seed = ss))
        snn.init()        
        res = zeros(NO,NO)
        if snn.batch_size>len(ytest):
            bb = len(ytest)
        else:
            bb = snn.batch_size
            
        # get data
        for xlocal,ylocal in ListHauler(xtest,ytest,snn.batch_size,3000):
            snn.run(xlocal)
            out = (snn.output.max(dim = 1).values.argmax(dim = 1))
            for jj in range(snn.batch_size):
                res[ylocal[jj],int(out[jj])] +=1
        # calculate accuracy
        ac_array[kk,zz] = (res*eye(NO)).sum()/res.sum()
        res_matrix[kk,:,:] = res
 
##### plot check ##############################################################
    
f,(a1,a2,a3) = plt.subplots(ncols = 3, nrows = 1, dpi = 300,figsize=(18*cm, 5*cm),width_ratios=[1, 1, 1]);
a1.plot(arange(len(data['test_ac']))*5,data['training_ac'],'-',color = [0,0,0],label = 'training')
a1.plot(arange(len(data['test_ac']))*5,data['test_ac'],'-',color = [.8,0,0],label = 'test')
a1.legend()
a1.set_xticks(arange(0,400,step = 100).numpy())
a1.set_xlabel("epochs")
a1.set_ylabel("accuracy [abs.]")
res = data['cm']
mp = a2.imshow(res/res.sum(dim=1),cmap = "jet",clim =[0,1]);
a2.set_xticks(arange(NO).numpy(),labels = {'⠁','⠃','⠉','⠙','⠑','⠋','⠛','⠓','⠊','⠚'})
a2.set_yticks(arange(NO).numpy(),labels = {'⠁','⠃','⠉','⠙','⠑','⠋','⠛','⠓','⠊','⠚'})
a2.set_xlabel("prediction")
a2.set_ylabel("target")
plt.colorbar(mappable=mp, label = "accuracy")
a3.errorbar(investigated_levels,ac_array.mean(dim=1),ac_array.std(dim=1),color = [0.1,0.1,0.8],label = 'quantized')
a3.plot(investigated_levels,investigated_levels*0+ac_array[0,0],'--',color = [0,0,0.0], label = 'float precision')
a3.set_xlabel("number of levels")
a3.set_ylabel("accuracy [abs.]")
a3.set_xscale('log')
a3.set_xlim([investigated_levels[0].item(),investigated_levels[-1].item()])
a3.legend()
f.tight_layout()  

