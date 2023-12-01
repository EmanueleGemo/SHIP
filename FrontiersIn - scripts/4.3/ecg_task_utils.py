import wfdb # "A library of tools for reading, writing, and processing WFDB signals and annotations."
from numpy import (arange as rng,
                   concatenate,
                   zeros as npzeros)
from torch import (zeros,
                   arange,
                   long,
                   randperm,
                   cat,
                   div,
                   stack,
                   tensor,
                   is_tensor,
                   exp,
                   rand,
                   eye,
                   manual_seed)
from SHIP import (network,
                  list_inputN,
                  refractory_variabletimestep,
                  lifN,
                  lS_1o,
                  liN,
                  SurrGradTrainer as trainer,
                  ListHauler)
from scipy.interpolate import Akima1DInterpolator as akima
import pickle

def get_mitbih_data(buffer = 50,checked_classes = None, path = ''):
    
    if checked_classes is None:
        checked_classes = [ #"·", # Normal beat
                            "N", # Normal beat
                            "L",	#Left bundle branch block beat
                            "R",	#Right bundle branch block beat
                            "A",	#Atrial premature beat
                            "a",	#Aberrated atrial premature beat
                            "J",	#Nodal (junctional) premature beat
                            "S",	#Supraventricular premature beat
                            "V",	#Premature ventricular contraction
                            "F",	#Fusion of ventricular and normal beat
                            # "[",	#Start of ventricular flutter/fibrillation
                            "!",	#Ventricular flutter wave
                            # "]",	#End of ventricular flutter/fibrillation
                            "e",	#Atrial escape beat
                            "j",	#Nodal (junctional) escape beat
                            "E",	#Ventricular escape beat
                            "/",	#Paced beat
                            "f",	#Fusion of paced and normal beat
                            "x",	#Non-conducted P-wave (blocked APB)
                            "Q",	#Unclassifiable beat
                            "|" ]   #Isolated QRS-like artifact
    
    #                N/.	L	R	A	a	J	S	V	F	!	e	j	E	P	f	p	Q
    nums = ["100",#	2239	-	-	33	-	-	-	1	-	-	-	-	-	-	-	-	-
            "101",#	1860	-	-	3	-	-	-	-	-	-	-	-	-	-	-	-	2
            "102",#	99	-	-	-	-	-	-	4	-	-	-	-	-	2028	56	-	-
            "103",#	2082	-	-	2	-	-	-	-	-	-	-	-	-	-	-	-	-
            "104",#	163	-	-	-	-	-	-	2	-	-	-	-	-	1380	666	-	18
            "105",#	2526	-	-	-	-	-	-	41	-	-	-	-	-	-	-	-	5
            "106",#	1507	-	-	-	-	-	-	520	-	-	-	-	-	-	-	-	-
            "107",#	-	-	-	-	-	-	-	59	-	-	-	-	-	2078	-	-	-
            "108",#	1739	-	-	4	-	-	-	17	2	-	-	1	-	-	-	11	-
            "109",#	-	2492	-	-	-	-	-	38	2	-	-	-	-	-	-	-	-
            "111",#	-	2123	-	-	-	-	-	1	-	-	-	-	-	-	-	-	-
            "112",#	2537	-	-	2	-	-	-	-	-	-	-	-	-	-	-	-	-
            "113",#	1789	-	-	-	6	-	-	-	-	-	-	-	-	-	-	-	-
            "114",#	1820	-	-	10	-	2	-	43	4	-	-	-	-	-	-	-	-
            "115",#	1953	-	-	-	-	-	-	-	-	-	-	-	-	-	-	-	-
            "116",#	2302	-	-	1	-	-	-	109	-	-	-	-	-	-	-	-	-
            "117",#	1534	-	-	1	-	-	-	-	-	-	-	-	-	-	-	-	-
            "118",#	-	-	2166	96	-	-	-	16	-	-	-	-	-	-	-	10	-
            "119",#	1543	-	-	-	-	-	-	444	-	-	-	-	-	-	-	-	-
            "121",#	1861	-	-	1	-	-	-	1	-	-	-	-	-	-	-	-	-
            "122",#	2476	-	-	-	-	-	-	-	-	-	-	-	-	-	-	-	-
            "123",#	1515	-	-	-	-	-	-	3	-	-	-	-	-	-	-	-	-
            "124",#	-	-	1531	2	-	29	-	47	5	-	-	5	-	-	-	-	-
            "200",#	1743	-	-	30	-	-	-	826	2	-	-	-	-	-	-	-	-
            "201",#	1625	-	-	30	97	1	-	198	2	-	-	10	-	-	-	37	-
            "202",#	2061	-	-	36	19	-	-	19	1	-	-	-	-	-	-	-	-
            "203",#	2529	-	-	-	2	-	-	444	1	-	-	-	-	-	-	-	4
            "205",#	2571	-	-	3	-	-	-	71	11	-	-	-	-	-	-	-	-
            "207",#	-	1457	86	107	-	-	-	105	-	472	-	-	105	-	-	-	-
            "208",#	1586	-	-	-	-	-	2	992	373	-	-	-	-	-	-	-	2
            "209",#	2621	-	-	383	-	-	-	1	-	-	-	-	-	-	-	-	-
            "210",#	2423	-	-	-	22	-	-	194	10	-	-	-	1	-	-	-	-
            "212",#	923	-	1825	-	-	-	-	-	-	-	-	-	-	-	-	-	-
            "213",#	2641	-	-	25	3	-	-	220	362	-	-	-	-	-	-	-	-
            "214",#	-	2003	-	-	-	-	-	256	1	-	-	-	-	-	-	-	2
            "215",#	3195	-	-	3	-	-	-	164	1	-	-	-	-	-	-	-	-
            "217",#	244	-	-	-	-	-	-	162	-	-	-	-	-	1542	260	-	-
            "219",#	2082	-	-	7	-	-	-	64	1	-	-	-	-	-	-	133	-
            "220",#	1954	-	-	94	-	-	-	-	-	-	-	-	-	-	-	-	-
            "221",#	2031	-	-	-	-	-	-	396	-	-	-	-	-	-	-	-	-
            "222",#	2062	-	-	208	-	1	-	-	-	-	-	212	-	-	-	-	-
            "223",#	2029	-	-	72	1	-	-	473	14	-	16	-	-	-	-	-	-
            "228",#	1688	-	-	3	-	-	-	362	-	-	-	-	-	-	-	-	-
            "230",#	2255	-	-	-	-	-	-	1	-	-	-	-	-	-	-	-	-
            "231",#	314	-	1254	1	-	-	-	2	-	-	-	-	-	-	-	2	-
            "232",#	-	-	397	1382	-	-	-	-	-	-	-	1	-	-	-	-	-
            "233",#	2230	-	-	7	-	-	-	831	11	-	-	-	-	-	-	-	-
            "234"]# 2700	-	-	-	-	50	-	3	-	-	-	-	-	-	-	-	-
    
    x = []
    y = []
    
    for num in nums:
    
        try:
            record = wfdb.rdsamp(path+'mit-bih/'+num, m2s = False)
            # these are the signals, to be converted via analog2spike function
            channels = record.p_signals
        except:
            channels,_ = wfdb.rdsamp(path+'mit-bih/'+num)
        
        
        annotation = wfdb.rdann(path+'mit-bih/'+num, 'atr')
        # these contain the labels, which need to be cleaned up, keeping only the checked classes
        labels = annotation.symbol
        tlabels = annotation.sample
        l = len(annotation.symbol)
        
        #cleanup:
        #  remove dots if existent
        labels = ["N" if (labels[ii]=="·" or labels[ii]==".") else labels[ii] for ii in rng(l)]  
        #  get only cheched classes
        mask = [True if label in checked_classes else False for label in labels]
        
        # plt.plot(tensor(tlabels[mask]).diff())
        # plt.title(num)
        # plt.show()
    
        #  keep only desired stuff
        labels = [labels[ii] for ii in rng(l) if mask[ii]]
        tlabels = tlabels[mask]
        
        # split data in chunks    
        # normal_labels = array([0 if _ == "N" else 1 for _ in labels])
        ti = concatenate((npzeros(1),(tlabels[1:]+tlabels[:-1])//2),axis = 0).astype(int) - buffer#normal_labels*buffer
        ti[0] = 0
        te = concatenate(((tlabels[1:]+tlabels[:-1])//2,npzeros(1)),axis = 0).astype(int) + buffer#normal_labels*buffer
        te[-1] = len(channels)
        
        for i,e,ll,tt in zip(ti,te,labels,tlabels):
            x.append(tensor(channels[i:e]))
            y.append(checked_classes.index(ll))
        
    y = tensor(y)
    return x,y

def get_encoding(bits): # utility for the delta encoding algorithm
    mid = (2**(bits+1)-1)//2
    inputs = 2*bits
    code = zeros((2**(bits+1)-1,2*bits), dtype = bool)
    code[mid:,:bits]  = stack([ tensor(_).unsqueeze(-1).bitwise_and( 2**arange(bits) ).ne(0).bool()   for _ in range(2**bits)], dim = 0)
    code[:mid+1,bits:] = ~code[mid:,:bits]
    return mid,inputs,code

def analog2spike(aa,
                 bits = 1,
                 dt = 0.001,
                 threshold=0.005,
                 sr = 360  ):

    """
    aa is a list of analog signals (each signal may contain data from multiple
    channels)
    bits is the precision by which the data is encoded (1 by default)
        note - encoding is carried out as follows (example with 2 bits):
        [2**0 2**1 -2**0 -2**1]
    dt is the planned timestep size
    max_rate is the maximum firing rate with which the signal is encoded
    sr is the signal resolution
    """
    
    mid,inputs,code = get_encoding(bits)
    
    spiking_input = []
    
    if not isinstance(aa,list):
        aa = [aa]

    print('converting analog data to spike traces:')
    out = []
     
    for i,a in enumerate(aa): 
        print(i)           
        if not is_tensor(a):
            a = tensor(a)
            
        data_nts = a.shape[0]
        
        if a.dim()==1:
            channels = 1
        else:
            channels = a.shape[1]
            
        data_rng = rng(data_nts)
        interp_rng = rng(data_nts/sr//dt)
        ln = interp_rng.shape[0]
        interp_rng = interp_rng/interp_rng[-1]*data_rng[-1]
        
        interp = tensor( akima( data_rng, a.numpy()) ( interp_rng ) )
        
        spiking_input = zeros((ln,inputs*channels),dtype = bool)   
                       
        dc = zeros(channels,dtype = a.dtype)
        for kk,row in enumerate(interp):
            
            idx = div(row-dc,threshold).to(int).clip(-mid,mid)
            mp = idx>0
            md = idx<0
            idx += mid
            if mp.any():
                dc[mp] = row[mp]
            if md.any():
                dc[md] = row[md]
            spiking_input[kk] = code[idx].view(-1)
        out.append(spiking_input)
    
    return out, {"dt": dt, "threshold": threshold, "sr": sr}

def get_datasets(fname = None,                   
                 checked_classes = None,
                       
                 bits = 1,
                    
                 buffer = 0, 
                 dt = 1e-3,
                 threshold=0.01,
                 sr = 360,
                    
                 train_dataset_maxsize = None,
                 test_dataset_maxsize = 100,
                    
                 shuffle = True,# if True the code shuffles the dataset before sampling
                 randomize = True, # if True, the code returns a randomized set obtained from the dataset; if False, the dataset will contain ordered samples (by label)
                 force_rebuild = False, # if True, it forces the code to rebuild the dataset
                    
                 path = ''):
    
           
    try: # file might be already present (long process - let's do it once)
        
        if fname is None and not force_rebuild:
            fname = "mitbih_%0.0fbit_sr%0.0fHz_buf%0.0fsteps_dt%0.2fms_th%0.3f_ts(%0.0f,%0.0f).pic"%(bits,sr,buffer,1000*dt,threshold,train_dataset_maxsize,test_dataset_maxsize)
        import pickle
        with open(fname,'rb') as handle:
            (x_train,y_train,  x_test,y_test, train_idx,test_idx, checked_classes,specs) = pickle.load(handle)
    
    except: # no file
        
        x,y = get_mitbih_data(buffer, checked_classes = checked_classes, path=path)
        
        cc = y.max().item()+1 # faster than: # y.unique().shape[0]# number of classes
        ss = y.shape[0] # number of samples

        if train_dataset_maxsize is None:
            train_dataset_maxsize = ss
        
        counts = zeros(cc,dtype = int)
        mp = zeros(ss,cc,dtype = bool)
        
        pp = arange(ss)
        if shuffle:
            pp = pp[randperm(ss)]
            
        test_size = zeros(cc,dtype = long)
        train_size = zeros(cc,dtype = long)
        train_idx = zeros(0,dtype = long)
        test_idx = zeros(0,dtype = long)
        for ii in range(cc):
            mp = y[pp]==ii
            counts[ii] = mp.count_nonzero()
            quarter_counts = counts[ii]//4+1
            compensate = quarter_counts<test_dataset_maxsize
            test_size[ii] = (~compensate)*test_dataset_maxsize + compensate*quarter_counts
            train_size[ii] = counts[ii]-test_size[ii]
            if train_size[ii]>train_dataset_maxsize:
                train_size[ii]= train_dataset_maxsize
            train_idx = cat( (train_idx, pp[mp][:train_size[ii]]), dim=0)
            test_idx = cat( (test_idx, pp[mp][(counts[ii]-test_size[ii]):]), dim=0)
            
        x_train,specs = analog2spike([x[_] for _ in train_idx], bits,dt,threshold,sr)
        y_train = y[train_idx]
        
        
        x_test,_ = analog2spike([x[_] for _ in test_idx], bits,dt,threshold,sr)     
        y_test = y[test_idx]

        import pickle
        fname = "mitbih_%0.0fbit_sr%0.0fHz_buf%0.0fsteps_dt%0.2fms_th%0.3f_ts(%0.0f,%0.0f).pic"%(bits,sr,buffer,1000*dt,threshold,train_dataset_maxsize,test_dataset_maxsize)
        with open(fname,'wb') as handle:
            pickle.dump((x_train,y_train,  x_test,y_test, train_idx,test_idx, checked_classes,specs), handle)
        
    return x_train,y_train,  x_test,y_test, train_idx,test_idx, checked_classes,specs

def minipillar(sz,imap,w=[3,6,-2,-2],k=[.45,.3,.6,.15],l = 2):
    
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
    P = zeros(N,N)
    x = arange(sz[0]).unsqueeze(-1).unsqueeze(-1).expand(sz).reshape(N)
    y = arange(sz[1]).unsqueeze(0).unsqueeze(-1).expand(sz).reshape(N)
    z = arange(sz[2]).unsqueeze(0).unsqueeze(0).expand(sz).reshape(N)
    
    ss = arange(N).unsqueeze(-1).expand(N,N).reshape(N**2)
    tt = arange(N).unsqueeze(0).expand(N,N).reshape(N**2)
    
    iss = imap.unsqueeze(-1).expand(N,N)
    itt = imap.unsqueeze(0).expand(N,N)
    kloc = tensor(k).unsqueeze(0)[:,2*iss+1*itt][0] 
    P = (kloc[ss,tt] * exp(-(  (x[ss]-x[tt]).pow(2)+(y[ss]-y[tt]).pow(2)+(z[ss]-z[tt]).pow(2)  ).abs()/l2)).reshape(N,N)  
    
    wloc = tensor(w).float().unsqueeze(0)[:,2*iss+1*itt][0]
    
    wloc[P<=rand(N,N)] = 0

    return wloc

def limited_connections(Ns,Nt,C = 4):
    # this function makes a w map that, given a synapse group of Ns x Nt elements,
    # and given a Nt-sized Imap distribution of inhibitory neurons, link all
    # neurons of the source to Ce - excitatory neurons of the target (and Ci
    # inhibitory neurons of the target)

    w = zeros(Ns,Nt)
    
    for jj in range(Ns):
        w[jj,randperm(Nt)[:C]] +=1    
    return w

def build_network(dt,batch_size,
                  nn, # iteration number
                  NI,Nr_dim,
                  bits,
                  IEratio,                  
                  refr_time,
                  tau_alpha,tau_beta,
                  reservoir_tau_beta,
                  reservoir_threshold,
                  w_reservoir_scaling_factor,
                  w_input_scaling_factor,
                  output_count,
                  seed = None):  
    
    if seed:
        manual_seed(seed)
         
    Nr = tensor(Nr_dim).prod().item()  
    
    imap = rand(Nr)<IEratio 
    ta = 1e-3*(4*imap.unsqueeze(1).expand(imap.shape[0],imap.shape[0])+4)
    input_reservoir_fn = lambda _,__: 8*2*(rand(_,__)-1/2)*limited_connections(_, __,4)
    w_reservoir = minipillar(Nr_dim,imap)
    i_reservoir = input_reservoir_fn(4,Nr)    
    i_reservoir_ = zeros(i_reservoir.shape[0]*bits,i_reservoir.shape[1])
    for jj in range(0,bits):
        i_reservoir_[slice(jj,i_reservoir.shape[1],bits),:] = (2**jj)*i_reservoir
 
    snn = network()
     
    snn.add(list_inputN,'i', N = NI)   
     
    snn.add(refractory_variabletimestep(lifN),'a',
            N = Nr, 
            _u_ = 0.,
            u0 = 0.,
            thr = reservoir_threshold, 
            tau_beta = reservoir_tau_beta,
            refr_time = refr_time)
    snn.add(lS_1o,'aa',source = 'a',target = 'a', 
            _I__ = 0,
            tau_alpha = ta, 
            w_scale = w_reservoir_scaling_factor, 
            w = w_reservoir)
    
    snn.add(liN,'p',N=output_count, _u_ = 0., u0 = 0., tau_beta = tau_beta, is_output = True)
    
    snn.add(lS_1o,'ia',source = 'i',target = 'a', 
            tau_alpha = tau_alpha, 
            w_scale = w_input_scaling_factor, 
            w = i_reservoir_)
 
    snn.add(lS_1o,'ap',source = 'a',target = 'p', 
            tau_alpha = tau_alpha, 
            w_scale = 30,
            _I__ = 0,
            w__ = rand)  

    snn.init(dt = dt, batch_size = batch_size)
    
    # backup of net params
    fname = "ecg_%0.0fbit_I%0.0f_net_parameters.pic"%(bits,nn)
    with open(fname,'wb') as handle:
        data = {'nn':nn, # iteration number
                'NI':NI,
                'Nr_dim':Nr_dim,
                'bits':bits,
                'IEratio':IEratio,                  
                'refr_time':refr_time,
                'tau_alpha':tau_alpha,
                'tau_beta':tau_alpha,
                'reservoir_tau_beta':reservoir_tau_beta,
                'reservoir_threshold':reservoir_threshold,
                'w_reservoir_scaling_factor':w_reservoir_scaling_factor,
                'w_input_scaling_factor':w_input_scaling_factor,
                'i_reservoir_':i_reservoir_,
                'w_reservoir':w_reservoir,
                'output_count':output_count,
                'seed':seed}
        pickle.dump(data,handle)    
    
    return snn

def one_run(dt,
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
            seed,
            
            xtrain,ytrain,
            xtest,ytest,
            epochs):
    
    snn = build_network(dt,
                        batch_size,
                        nn,
                        NI,Nr_dim,
                        bits,
                        IEratio,                  
                        refr_time,
                        tau_alpha,tau_beta,
                        bb,tt,ww,ii,
                        output_count,
                        seed = seed)
    
    trn = trainer(snn,trainable_synapse = 'ap')
    trn.init()
    
    # snn.set_monitor(**{'ia':['I'],'a':['u','output'],'ap':['I']})
    # core.batch_size = 10
    
    for ee in range(epochs):
        print('epoch no. %0.0f'%ee)
        for xlocal,ylocal in ListHauler(xtrain,ytrain,snn.batch_size,max_batches = len(ytrain)):  
            snn.run(xlocal)
            trn.run(labels = ylocal)      
            mean_ac = (snn.output.max(dim = 1).values.argmax(dim = 1) == ylocal).float().mean()
            print('loss = %0.3f, accuracy = %0.1f'%( trn.loss_hist[-1], mean_ac ) )

    # here getting accuracy with test dataset
    res = zeros(output_count,output_count)
    jj = 0
    ttl = 0
    for xlocal,ylocal in ListHauler(xtest,ytest,snn.batch_size,max_batches = len(ytest)):
        snn.run(xlocal)
        ttl = ttl + len(xlocal)
        out = (snn.output.max(dim = 1).values.argmax(dim = 1))
        for kk in range(snn.batch_size):
            res[ylocal[kk],int(out[kk])] +=1
        print('%0.0f..'%jj); 
        jj+=1
    
    final_accuracy = (res*eye(output_count)).sum()/ttl

    # save backup of trained weights and training history
    fname = ('ecg_%0.0fbit_I%0.0f_beta%0.3f_thr%0.02f_wr%0.1f_wi%0.1f.pic'%(bits,nn,bb,tt,ww,ii))
    with open(fname,'wb') as handle:
        pickle.dump({'w':snn.groups.ap.w,
                     'loss_hist':trn.loss_hist,
                     'confusion_matrix':res,
                     'accuracy':final_accuracy},handle)
    
    print('post-training accuracy for %s: %0.3f'%(fname,final_accuracy))
    return snn,final_accuracy

def get_network_behavior(snn,xtest,ytest):
    # snn.set_monitor(**{'ia':['I'],'a':['u','output'],'ap':['I']})
    snn.set_monitor(**{'a':['output'],'ap':['I']})
    sp = 0
    cu = 0
    for xlocal,ylocal in ListHauler(xtest,ytest,snn.batch_size,max_batches = len(ytest)):
        snn.run(xlocal)
        data = snn.get_monitored_results()
        sp = sp + data.a.output.sum() #<-sum of all spikes out of all reservoir neurons, along the time axis, for all batches
        cu = cu + data.ap.I.sum() #<-sum of all currents along all synapses, along the time axis, for all batches
        
    return sp/len(ylocal),cu/len(ylocal) # returning the average number of spikes per sample / synaptic current integral per sample