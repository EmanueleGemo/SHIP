"""
gathering the accuracy as a function of time (changing due to memristive drift)
"""

from SHIP import TensorHauler,network,liN,inputN
from braille_task_utils import (load_braille_dataset,
                                delta_encoding,
                                select_datasets)
from torch import (tensor,
                   arange,
                   zeros,
                   normal,
                   manual_seed,
                   eye,
                   zeros_like,
                   cat)
from braille_task_utils import izhikevicN,ilS_2o
import os,pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# global params
cm = 1/2.54  # centimeters in inches <- plot purposes
dt = 0.0005
nts = 300
batch_size = 16
fname = "results_braille_63.5.pic" #<-- we arbitrary select a savefile (@300 epochs)

# drift model derived from data in 10.1002/aisy.202200145
# functioning proposed on the basis of the schematics there illustrated
class ilS_2o_drift(ilS_2o):
    """
    integrating the drift functionality, updating the set_initial_state method
    """    
    parameters = ilS_2o.parameters | {'w0': None, #<-- as-written weight (saved in set_initial_state)
                                     'T':0} #<-- time elapsed from writing
    @staticmethod
    def drift_fun(w,t,seed = 0):
        """
        model from matlab:
        log10t = log10(t);
        sm0 = 0.005127+0.85543./(w0-38.72);
        sel = w0<64.41;
        mu = sel*((0.2302*w0-15.24)*log10t+0.9546*w0+4.82)+(1-sel)*(-0.4129*log10t+w0);
        smu = (1.047*sm0-0.006294)*log10t-0.01677*sm0+0.005473;
            
        range of applicability: t ~> 0.5 s (0.83s with the current parameters)
        range of reliabilty: 10 s ~< t < 10000
        """
        if t>0.83:
            manual_seed(seed)
            Cmax = 120
            Cmin = 50
            C0 = Cmin
            Cdiff = Cmax-Cmin
            # finding absolute max value M
            M = max(w.min().abs(),w.max().abs()).abs()
            # normalizing each w value to the differential conductance, and then shifting to the min-max uS range
            sign = w.sign()
            w0 = w.abs()/M*Cdiff+Cmin   
            # model: finding value of high-conductance memristor, subtracting value of low-conductance memristor 
            # (swapped for higher stability of high_conductance state)
            log10t = t.log10()
            # high
            sm0 = 0.005127+0.85543/(w0-38.72)
            sel = (w0<64.41).float()
            mu = sel*((0.2302*w0-15.24)*log10t +0.9546*w0+4.82) + (1-sel)*(-0.4129*log10t+w0);
            smu = (1.047*sm0-0.006294)*log10t -0.01677*sm0+0.005473;
            # low
            sm0_ = 0.005127+0.85543/(C0-38.72)
            sel_ = float(C0<64.41)
            mu_ = sel_*((0.2302*C0-15.24)*log10t +0.9546*C0+4.82) + (1-sel_)*(-0.4129*log10t+C0);
            smu_ = (1.047*sm0_-0.006294)*log10t -0.01677*sm0_+0.005473;
            # calculate differential conductance
            wdiff = normal(mean=zeros_like(mu)+mu_, std =smu_*mu_)-normal(mean = mu, std = smu*mu)
            # rescaling differential to in-platform range
            out = sign*(C0-wdiff-Cmin)/Cdiff*M   
            return out
        else:
            return w
    def set_initial_state(self, *args, T=0, seed = 0):
        super().set_initial_state(*args)
        if self.w0 is None: 
            self.w0 = self.w #<-initial storage
        if T>self.T: # temporal advancement
            self.T = T
            self.w = self.drift_fun(self.w0,T,seed = seed)
    
def build_datasets():
    # here copy-pasting from main (lazy - convenient)
    # main parameters    
    encoding_params = {"nts":nts,
                       "bits":1,
                       "delta":0.01,
                       "interp":10,
                       "swap":True}
    test_train_ratio = .2
    # get and convert dataset (delta-rule) 
    folder = os.getcwd()+"//"
    filename = "braille_dset.pic"
    x,y = load_braille_dataset(folder,filename)
    xe = delta_encoding(x,**encoding_params)
    xT,yT,xt,yt = select_datasets(xe,y,test_train_ratio = test_train_ratio,
                                  limitclasses = 10, seed = 150)
    
    return xt,yt
def build_network(fname,xt,yt):
    
    # network building
    Nrec = 125
    Nhid = 128
    classes = 1+yt.max()   
    datashape = xt.shape[2]
    n_model = izhikevicN
    s_model = ilS_2o_drift    
    manual_seed(200)

    # loading saved parameters
    with open (fname, "rb") as h:
        data = pickle.load(h)
        
    net = network()     
    net.add(inputN,'I',N = datashape)
    net.add(n_model,'L0',N = Nrec)
    net.add(n_model,'L1',N = Nhid)
    net.add(liN,'out',N = classes, is_output = True)
    net.add(s_model,'S0',source='I',target = 'L0',w_scale = 1000,w = data['w_S0'])
    net.add(s_model,'S1',source='L0',target = 'L1',w_scale = 800,w = data['w_S1'])
    net.add(s_model,'S2',source='L0',target = 'L0',w_scale = 1500,w = data['w_S2'])
    net.add(s_model,'S3',source='L1',target = 'out',w_scale = 100,w = data['w_S3'])

    net.init(dt = dt, nts = nts, batch_size = batch_size)
    return net,classes,data

##### main ####################################################################
    
xt,yt = build_datasets()
net,output_count,data = build_network(fname,xt,yt)

investigated_times = cat((tensor(0).unsqueeze(-1),tensor(10).pow(arange(0,4.1,0.5))),0)
seeds = arange(13,33).tolist()

ac_array = zeros(len(investigated_times),len(seeds))

for zz,ss in enumerate(seeds):
    res_matrix = zeros(len(investigated_times),output_count,output_count)
    for kk,t in tqdm(enumerate(investigated_times)):    
        # resetting network and local results for new run
        net.init()
        res = zeros(output_count,output_count)
        if net.batch_size>len(yt):
            bb = len(yt)
        else:
            bb = net.batch_size
        for xlocal,ylocal in TensorHauler(xt, yt,bb,3000):
            # net.set_monitor(**{'S2':['output']})
            net.run(xlocal,T=t,seed = ss)
            # data = net.get_monitored_results().S2.output[0,:,:]
            out = (net.output.max(dim = 1).values.argmax(dim = 1))
            for jj in range(net.batch_size):
                res[ylocal[jj],int(out[jj])] +=1
        ac_array[kk,zz] = (res*eye(output_count)).sum()/res.sum()
        res_matrix[kk,:,:] = res
 
##### plot check ##############################################################
    
investigated_times[0] = 0.5 # plot purpose (the model is nor reliable at t~<0.5s)
f,(a1,a2,a3) = plt.subplots(ncols = 3, nrows = 1, dpi = 300,figsize=(18*cm, 5*cm),width_ratios=[1, 1, 1]);
a1.plot(arange(len(data['test_ac']))*5,data['training_ac'],'-',color = [0,0,0],label = 'training')
a1.plot(arange(len(data['test_ac']))*5,data['test_ac'],'-',color = [.8,0,0],label = 'test')
a1.legend()
a1.set_xticks(arange(0,400,step = 100).numpy())
a1.set_xlabel("epochs")
a1.set_ylabel("accuracy [abs.]")
res = data['cm']
mp = a2.imshow(res/res.sum(dim=1),cmap = "jet",clim =[0,1]);
a2.set_xticks(arange(output_count).numpy(),labels = {'⠁','⠃','⠉','⠙','⠑','⠋','⠛','⠓','⠊','⠚'})
a2.set_yticks(arange(output_count).numpy(),labels = {'⠁','⠃','⠉','⠙','⠑','⠋','⠛','⠓','⠊','⠚'})
a2.set_xlabel("prediction")
a2.set_ylabel("target")
plt.colorbar(mappable=mp, label = "accuracy")
a3.errorbar(investigated_times,ac_array.mean(dim=1),ac_array.std(dim=1),color = [0.1,0.1,0.8],label = 'f(t)')
a3.plot(investigated_times,investigated_times*0+ac_array[0,0],'--',color = [0,0,0.0], label = 't=0')
a3.set_xlabel("time [s]")
a3.set_ylabel("accuracy [abs.]")
a3.set_xscale('log')
a3.set_xlim([0.5,11000])
a3.legend()
f.tight_layout()  

