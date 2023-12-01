from torch import (tensor,
                   is_tensor,
                   cat,
                   zeros,
                   arange,
                   exp,
                   rand,
                   randperm,
                   manual_seed,
                   bool as tbool,
                   eye,
                   ones,
                   stack)
from SHIP import (network, 
                  lS_1o,
                  lifN,
                  liN,
                  list_inputN,
                  SurrGradTrainer as trainer,
                  ListHauler,
                  refractory_variabletimestep as refractory,
                  delayed)
import pickle
from tqdm import tqdm

# dataset handling
def wav_2_LyonCoch(wav,sample_rate=8000,decimation_factor = 32,ear_q = 8, 
                   step_factor = None, differ = None, agc = True, tau_factor = 3, dt = 0.001):
    # converts wav analog inputs (list datatype) to a list of cochleogram arrays,
    # interpolated with an akima spline at the timesteps from 0 to array_size*dt
    
    # other dependencies
    if not "LyonCalc"in dir():
        from lyon.calc import LyonCalc #<-- required -- pip install lyon (works only on windows as far as I understand)
    if not "Akima"in dir():
        from scipy.interpolate import Akima1DInterpolator as Akima
    if not "rng" in dir():    
        from numpy import (arange as rng,
                           nan_to_num)
        
    # check input datatype
    if not isinstance(wav,list):
        wav = [wav]
    # convert-interpolate datasets
    times = tensor([len(w) for w in wav])/sample_rate
    output = []
    for i,w in enumerate(wav):            
        if is_tensor(w):
            w = w.numpy() # this module works with numpy arrays
        
        cochleogram = LyonCalc().lyon_passive_ear(w,
                         sample_rate=sample_rate,
                         decimation_factor=decimation_factor,
                         ear_q=ear_q,
                         step_factor=step_factor,
                         differ=differ,
                         agc=agc,
                         tau_factor=tau_factor)
        
        # interpolate results with an akima spline to retrieve required datapoints 
        interp = nan_to_num ( Akima( rng(cochleogram.shape[0]), cochleogram ) ( rng(cochleogram.shape[0],step = cochleogram.shape[0]/times[i]*dt) ) ,copy = False)       
        output.append(interp)        
    return output

def cochleogram_2_spikes(cochs,dt = 0.001, max_rate = 500,thr = True):
    # linear conversion from cochleogram intensity values to rate values;
    # then uses these rates to generate spikes (deterministically)
    # the thr value is an intensity threshold, below which no spike takes place
    
    l = []
    for c in cochs:
        c = tensor(c)
        l.append(c.max())
    avg = tensor(l).min() #<- find average intensity of dataset
    
    if thr is False:
        thr = 0
    elif thr is True:
        thr = 0.5

    avg_thr = (avg*thr).numpy() #<-- apply threshold to average
    d = [cat(( zeros(1,_.shape[1]), (_/avg*max_rate*dt*(_>avg_thr)).cumsum(0) )).div(1,rounding_mode = 'floor') for _ in cochs]
    x_list = [d[_][1:]>d[_][:-1] for _ in range(len(d))]
    return x_list

def get_spiking_input(dataset_fname,Lyon_params,deltaencoder_params,backup_dsetfname):
    
    if not 'pickle' in dir():
        import pickle
    with open(dataset_fname,'rb') as handle:
        data = pickle.load(handle)
        wav = data['x_list']
        labels = data['y_train'].astype('int64')
    # convert dataset using Lyon's model
    try:
        coch = wav_2_LyonCoch(wav,**Lyon_params)
    except: # Lyon module does not work on all operative systems
        with open(backup_dsetfname,'rb') as handle:
            coch = pickle.load(handle)  
    # convert cochleograms into spiking sets
    spks = cochleogram_2_spikes(coch,**deltaencoder_params)
    
    return spks,labels

def select_traintestdata(x,y, train_fraction = 0.85, 
                         test_idx = None, 
                         dsetfname = None,
                         seed = 0):
    
    manual_seed(seed)    
    if test_idx is None:
        train_idx = zeros(0)
        test_idx = zeros(0)
        for jj in range(10):
            idx = (y==jj).nonzero()
            te = tensor(idx.shape[0]*train_fraction).int().item()
            newsequence = idx[randperm(idx.shape[0])]
            
            train_idx = cat( (train_idx,newsequence[0:te]) ,0)                      
            test_idx = cat( (test_idx,newsequence[te:]),0)
    else:
        all_idx = arange(len(y))
        m = ones(len(y))
        m[test_idx.long()] = False
        train_idx = all_idx[m.bool()]
    
    manual_seed(seed)
    try:
        xtrain = [x[int(train_idx[_].item())][0,:,:] for _ in range(len(train_idx))]
        xtest = [x[int(test_idx[_].item())][0,:,:] for _ in range(len(test_idx))]        
    except:
        xtrain = [x[int(train_idx[_].item())][:,:] for _ in range(len(train_idx))]    
        xtest = [x[int(test_idx[_].item())][:,:] for _ in range(len(test_idx))]
        
    ytrain = stack([y[int(train_idx[_].item())] for _ in range(len(train_idx))],0).long()
    ytest = stack([y[int(test_idx[_].item())] for _ in range(len(test_idx))],0).long()   

    if dsetfname: # here we save the encoded dataset for future use
        with open(dsetfname, "wb") as h:
            pickle.dump((xtest,ytest,xtrain,ytrain,test_idx),h)
    return xtest,ytest,xtrain,ytrain, test_idx

# network building

def minipillar(sz,imap,w=[1.5,3,-.5,-.5],k=[.45,.3,.6,.15],l = 2):
    
    # function returning the weight map according to Maass' minipillar model
    
    if not isinstance(sz,list):
        sz = [sz]
    ln = len(sz)
    if ln<3:
        print('WARNING - the size of the reservoir must be provided in a 3D volume fashion (resizing the provided data)')
        if ln == 1:
            sz.append(1)
            ln+=1
        if ln == 2:
            sz.append(1)
            
    l2 = l**2
        
    N = imap.shape[0]
    # W = torch.zeros(N,N)
    P = zeros(N,N)
    x = arange(sz[0]).unsqueeze(-1).unsqueeze(-1).expand(sz).reshape(N)
    y = arange(sz[1]).unsqueeze(0).unsqueeze(-1).expand(sz).reshape(N)
    z = arange(sz[2]).unsqueeze(0).unsqueeze(0).expand(sz).reshape(N)
    
    ss = arange(N).unsqueeze(-1).expand(N,N).reshape(N**2)
    tt = arange(N).unsqueeze(0).expand(N,N).reshape(N**2)
    # kk = torch.zeros(N**2)
    
    iss = imap.unsqueeze(-1).expand(N,N)
    itt = imap.unsqueeze(0).expand(N,N)
    kloc = tensor(k).unsqueeze(0)[:,2*iss+1*itt][0] 
    P = (kloc[ss,tt] * exp(-(  (x[ss]-x[tt]).pow(2)+(y[ss]-y[tt]).pow(2)+(z[ss]-z[tt]).pow(2)  ).abs()/l2)).reshape(N,N)  
    
    wloc = tensor(w).float().unsqueeze(0)[:,2*iss+1*itt][0]
    
    wloc[P<=rand(N,N)] = 0

    return wloc

def build_conversion_net(NI,NR,thrR,tbR,refr_timeR,taI,taE,phiRR,delay_timeRR,phiIR,
                         Nr_dim,imap,C,seed = None, wI = 64,
                         dt = 0.001, nts = 300, batch_size = 16):
    
    if seed:
        manual_seed(seed)
    
    # build params based on imap
    lc = zeros(NI,NR)
    for jj in range(NI):            
        lc[jj,randperm(NR)[:C]] +=1 # connecting C random reservoir neurons per input
    wIR = wI*(1-2*imap.unsqueeze(0).expand(NI,NR))*lc 
    wRR = minipillar(Nr_dim,imap)    
    imap2d = imap.unsqueeze(-1).expand(NR,NR)    
    taRR = (wRR!=0)*(~imap2d*taE+imap2d*taI)
    
    # build net
    snn = network()
    snn.add(list_inputN,'i', N = NI)
    snn.add(refractory(lifN),'r',
            N = NR, 
            thr = thrR, 
            tau_beta = tbR,
            refr_time = refr_timeR,
            is_output = True)
    
    snn.add(delayed(lS_1o),'rr',source = 'r',target = 'r', 
            _I__ = 0,
            tau_alpha = taRR,
            w_scale = phiRR, 
            w = wRR,
            delay_time = delay_timeRR)
    snn.add(lS_1o,'ir',source = 'i',target = 'r', 
            tau_alpha = taE,
            w_scale = phiIR,
            w = wIR)
    snn.init(dt = dt, nts = nts, batch_size = batch_size)
    
    return snn


def build_trainable_net(NR,NO,tbO,taRO,phiRO,
                        s_model = lS_1o, w = None, seed = None,
                        dt = 0.001, nts = 300, batch_size = 16):
    
    if seed:
        manual_seed(seed)     
    if w is None:
        w = 1-2*rand(NR,NO) # uniform dist

    snn = network()
    snn.add(list_inputN,'i', N = NR)
    snn.add(liN,'o',N=NO, tau_beta = tbO, is_output = True)
    snn.add(s_model,'io',source = 'i',target = 'o', 
            tau_alpha = taRO, 
            w_scale = phiRO,
            w = w)
    snn.init(dt = dt, nts = nts, batch_size = batch_size)
    return snn

def reservoir_spikes(snn,x_list,y,batch_size): 
    # here we simulate inference only on the reservoir, catching its output
    ii = 0#core.batch_size
    spiking_input = []
    spiking_y = zeros(len(y))
    for xlocal,ylocal in tqdm(ListHauler(x_list,y,batch_size,max_batches = len(y)),total = len(y)/batch_size):
        ii+=snn.batch_size
        snn.run(xlocal)
        # store
        for jj in range(snn.batch_size):
            spiking_input.append(snn.output[jj,:,:])
        spiking_y[(ii-snn.batch_size):ii] = tensor(ylocal).long()        
    return spiking_input,spiking_y
    
def train_1_epoch(snn,
                  trn,
                  xtrain,ytrain,
                  xtest,ytest,
                  ee,
                  shuffle,
                  training_ac,
                  test_ac,
                  loss_hist,
                  noise=0.01,
                  save_cnt = 10,
                  results_fname = 'results_spokendigit'):
    
    local_training_ac = zeros(1)
    res = None

    for xlocal,ylocal in ListHauler(xtrain,ytrain,snn.batch_size,max_batches = 3000,shuffle=shuffle):  
        
        if noise:
            for xx in xlocal:
                m = rand(xx.shape[0],xx.shape[1])<noise
                xx[m] = ~xx[m] if xx.dtype is tbool else 1-xx[m]
        
        snn.run(xlocal)
        trn.run(labels = ylocal)
        # here retrieving output layer results to track the accuracy        
        local_training_ac = local_training_ac + (snn.output.max(dim = 1).values.argmax(dim = 1) == ylocal).sum()

    local_training_ac = local_training_ac/len(ytrain)
    status = "current accuracy: training dset = %0.3f"%local_training_ac.item()
    
    # if test accuracy is also needed:
    if (ee+1)%save_cnt ==0:
        NO = ytest.max()+1
        res = zeros(NO,NO)
        ii = 0
        for xlocal,ylocal in ListHauler(xtest,ytest,snn.batch_size,3000):
            snn.run(xlocal)
            out = (snn.output.max(dim = 1).values.argmax(dim = 1))
            for jj in range(snn.batch_size):
                res[ylocal[jj],int(out[jj])] +=1 
            ii+=1
        local_test_ac = (res*eye(NO)).sum()/res.sum()            
        status = status + ", test dset = %0.3f\n"%local_test_ac.item()
        
        ### save results backup
        fname = results_fname+"_%0.1f.pic"%(local_test_ac.unsqueeze(-1).item()*100)
        with open(fname,'wb') as handle:            
            if loss_hist:
                training_ac = cat((training_ac,local_training_ac.unsqueeze(-1)),0)
                test_ac = cat((test_ac,local_test_ac.unsqueeze(-1)),0) 
                loss_hist.append(trn.loss_hist)
            else:
                training_ac = local_training_ac.unsqueeze(-1)
                test_ac = local_test_ac.unsqueeze(-1)
                loss_hist = trn.loss_hist  
            data = {'training_ac':training_ac,
                    'test_ac':test_ac,
                    'loss_hist':loss_hist,
                    'cm':res}
            for gg in snn.groups.values():
                if hasattr(gg,'w'):
                    data['w_%s'%gg.tag] = gg.w.clone().detach()
            pickle.dump(data,handle)
            
    return snn,trn,training_ac,test_ac,loss_hist,res,status