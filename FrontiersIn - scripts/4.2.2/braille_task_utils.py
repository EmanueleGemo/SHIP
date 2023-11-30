""" utils for the Izhikevic-network training task """

from SHIP import synapsegroup,neurongroup,TensorHauler
from SHIP.component_utils import get_alphas, quick_activator
from torch import (rand,
                   is_tensor,
                   tensor,
                   zeros,
                   cat,
                   searchsorted,
                   arange,
                   logical_and,
                   manual_seed,
                   randperm,
                   eye,
                   exp)
from tqdm import tqdm
import os
import pickle

class ilS_2o(synapsegroup):
    """
    Mimic of a 2nd order integrate-leaky synapse.
    """    
    variables = {'_I1_': 0,
                 '_I2_': 0}# synapse current contributions <--> internal states
    parameters = {'tau_alpha1_': 15e-3, # temporal constant [s]
                  'tau_alpha2_': 5e-3,                  
                  'w__': lambda s,t: -0.4+rand(s,t), # synaptic weight
                  'w_scale': 1} # scaling factor    
    # override the function determining the number of units
    def get_N(self):
        return self.Nt
    def time_dep(self):
       self.alpha1A,self.alpha1B = get_alphas(self.tau_alpha1,self.dt)
       self.alpha2A,self.alpha2B = get_alphas(self.tau_alpha2,self.dt)
    def set_initial_state(self, *args, **kwargs):
        if not is_tensor(self.w_scale): # data type checking
            self.w_scale = tensor(self.w_scale)
    def advance_timestep(self,local_input=zeros(1)):
        local_input = (local_input.unsqueeze(-1).expand(self.batch_size,self.Ns,self.Nt)*self.w*self.w_scale).sum(dim=1)        
        self.I1 = self.I1*self.alpha1A + local_input
        self.I2 = self.I2*self.alpha2A - local_input    
        out = self.I1*self.alpha1B+self.I2*self.alpha2B
        return out

class izhikevicN(neurongroup):
    '''
    Simple time-discrete deployment of the Izhikevic model.
    An additional parameter,"gain", multiplies the incoming input.
    This model can only broadcast single-valued parameters.
    The algorithm and default values of the variables and 
    parameters have been derived from:
        
    Izhikevich, E. M. (2003). Simple model of spiking neurons. 
    IEEE Transactions on neural networks, 14(6), 1569-1572. 
    10.1109/TNN.2003.8204
    '''
    
    variables = {'_v_': -65,
                 '_u_': 0}
    parameters = {'thr': 30,
                  'a':0.02,
                  'b':0.2,
                  'c': -65,
                  'd':8,
                  'p2':0.04,
                  'p1':5,
                  'p0':140}
    
    activator = quick_activator 
    
    def time_dep(self):
        """
        there is no dependency on the time-step size other than the 
        time-step size itself
        """
        self.P0 = self.p0*self.dt
        self.P1 = 1+self.p1*self.dt
        self.P2 = self.p2*self.dt
        self.Pu = 1-self.dt*self.a
        self.Pv = self.dt*self.a*self.b
        
    def set_initial_state(self, *args, **kwargs):
        self.u = self.b*self.v 
        pass

    def advance_timestep(self,local_input=0):        
        spikes = self.activator(self.v-self.thr)
        spiked = spikes.detach().bool()
        
        self.v[spiked] = self.c
        self.u[spiked] = self.u[spiked]+self.d
        self.v = self.v*(self.P2*self.v+self.P1)+self.P0-self.dt*self.u+local_input
        self.u = self.u*self.Pu + self.Pv*self.v
        return spikes

def load_braille_dataset(folder = os.getcwd()+"//",filename = "braille_dset.pic",force_rebuild = False):
    
    if not "Image" in dir():
        from PIL import Image
    if not "convert_tensor" in dir():
        from torchvision import transforms
        convert_tensor = transforms.ToTensor()
    if not "pickle" in dir():
        import pickle
    
    datafolder = folder+"Braille Dataset//"
        
    if filename not in os.listdir(folder) or force_rebuild:

        # convert jpegs into torch tensors
        allf = [_ for _ in os.listdir(+datafolder) if _[-4:] == ".jpg"]
        
        x = zeros(len(allf),28,28)
        for ii,f in enumerate(allf):
            img = Image.open(datafolder+f)
            x[ii] = convert_tensor(img)[0,:,:]
        
        y = zeros((len(allf)),dtype = int)
        for ii,f in enumerate(allf):
            y[ii] = ord(f[0])-97
        
        with open(folder+filename,'wb') as h:
            pickle.dump((x,y),h)
    else:
        with open(folder+filename,'rb') as h:
            x,y = pickle.load(h)            
    return x,y

def spline_interpolator(x,yy,xs):  
    """
    derived from: https://gist.github.com/chausies
    this algorithm:
    . assumes y to be a tensor of shape batch_size, i, j (possibly editable to be more flexible)
    . interpolates data along the j axis
    . assumes x as the axis coordinates along the j direction
    . and sees xs as the required interpolation coordinates
    """
    # poly
    A = tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
        ], dtype=yy[-1].dtype)    
    
    # central diff
    mm = (yy[:,:,1:] - yy[:,:,:-1])/(x[1:] - x[:-1])
    mm = cat([mm[:,:,[0]], (mm[:,:,1:] + mm[:,:,:-1])/2, mm[:,:,[-1]]],dim = -1)    
    I = searchsorted(x[1:], xs) # matching coordinates
    dx = (x[I+1]-x[I]) # unit distance from matching coords
    t = (xs-x[I])/dx # deltas between matching and forward closest unmatching coords
    tt = [ None for _ in range(4) ]
    tt[0] = 1
    for i in range(1, 4):
      tt[i] = tt[i-1]*t # calculating prefactors based on deltas 
    hh = [sum( A[i, j]*tt[j] for j in range(4) ) for i in range(4)]# prefactors      
    # return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx #interpolation
    return hh[0]*yy[:,:,I] + hh[1]*mm[:,:,I]*dx + hh[2]*yy[:,:,I+1] + hh[3]*mm[:,:,I+1]*dx

def delta_encoding(xx,delta = 0.1,nts = 300,bits = 1,interp = 5,swap = True):
    if interp != 1:
        h = arange(0,xx.shape[-1])
        k = arange(0,h[-1],1/interp)
        kl = len(k)
        xx = spline_interpolator(h,xx,k)
    else:
        kl = xx.shape[-1]
        
    if kl>nts:
        raise Exception('the number of timesteps is not sufficient to contain the interpolated data (increase nts above %0.0f)'%kl)
    
    out = zeros(xx.shape[0],xx.shape[1],2**bits,nts,dtype = bool)
    p = xx[:,:,0]
    bb = arange(-bits,bits+1);
    bb = bb[bb!=0]
    tl = arange(1,kl)
    ntstl = arange(0,nts)
    for tt in tl:
        deltas = xx[:,:,tt]-p
        spikes = (deltas.abs()//delta).clamp(min = 0, max = bits)
        reverse = logical_and(deltas<0,spikes>0)
        spikes[reverse] = -spikes[reverse]
        for ii in bb:
            where = spikes == ii;
            out[where,bb==ii,tt==ntstl] = True
                
        above = deltas>delta
        below = deltas<-delta
        if above.any():
            p[above] = xx[:,:,tt][above]
        if below.any():
            p[below] = xx[:,:,tt][below]
    
    if swap:
        return out.reshape(xx.shape[0],xx.shape[1]*2**bits,nts).swapaxes(1,2)
    else:
        return out.reshape(xx.shape[0],xx.shape[1]*2**bits,nts)
    
def select_datasets(x,y,test_train_ratio = .2,limitsamples = None, limitclasses = None, seed = None):
    if seed:
        manual_seed(seed)
    num_of_classes = y.max()+1
    if limitclasses and num_of_classes>limitclasses:
        num_of_classes = limitclasses
    num_of_samples = y.numel()
    howmanytrain = 0
    howmanytest = 0
    train_indices = zeros(num_of_samples,dtype = int)
    test_indices = zeros(num_of_samples,dtype = int)
    ii = 0
    jj = 0    
    for cc in arange(num_of_classes):
        mask = y==cc
        howmany = mask.count_nonzero()
        if limitsamples and howmany>limitsamples:
            howmany = limitsamples
        train_size = int(howmany*(1-test_train_ratio/(1+test_train_ratio)))
        test_size = howmany-train_size
        if test_size <1: # arbitrary: overlapping test and train
            test_size = test_size+1
        howmanytrain = howmanytrain+train_size
        howmanytest = howmanytest+test_size
        local_indices = arange(num_of_samples)[mask][randperm(howmany)]
        train_indices[ii:howmanytrain] = local_indices[:train_size]
        test_indices[jj:howmanytest] = local_indices[-test_size:]
        ii = howmanytrain
        jj = howmanytest
    return (x[train_indices[:howmanytrain],:],
            y[train_indices[:howmanytrain]],
            x[test_indices[:howmanytest],:],
            y[test_indices[:howmanytest]])

def recurrent_topology(sz,imap,w=[1,2,-2/3,-2/3],k=[.45,.3,.6,.15],l = 2):
    
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

def train1epoch(net,
                trn,
                xT,yT,
                xt,yt,
                ee,                  
                training_ac,
                test_ac,
                loss_hist,
                output_count,
                save_cnt = 10,
                results_fname = 'results_braille',
                max_batches = 3000,
                vocal = False):
    
    res = None
    status = None
    
    local_training_ac = zeros(1)    
    hwmn = 0
    if vocal:
        for xlocal,ylocal in tqdm(TensorHauler(xT, yT, batch_size = 16, max_batches = int(1e9),shuffle=True),total = yT.numel()//16):  
            hwmn += net.batch_size
            net.run(xlocal)
            trn.run(labels = ylocal)
            local_training_ac = local_training_ac + (net.output.max(dim = 1).values.argmax(dim = 1) == ylocal).sum()
        
    else:
        for xlocal,ylocal in TensorHauler(xT, yT, batch_size = 16, max_batches = int(1e9),shuffle=True):  
            hwmn += net.batch_size
            net.run(xlocal)
            trn.run(labels = ylocal)
            local_training_ac = local_training_ac + (net.output.max(dim = 1).values.argmax(dim = 1) == ylocal).sum()
        
    local_training_ac = local_training_ac/hwmn
    status = "current accuracy: training dset = %0.3f"%local_training_ac.item()
    
    if (ee+1)%save_cnt ==0:
        
        res = zeros(output_count,output_count)
        ii = 0
        if net.batch_size>len(yt):
            bb = len(yt)
        else:
            bb = net.batch_size
        for xlocal,ylocal in TensorHauler(xt, yt,bb,3000):
            net.run(xlocal)
            out = (net.output.max(dim = 1).values.argmax(dim = 1))
            for jj in range(net.batch_size):
                res[ylocal[jj],int(out[jj])] +=1
            ii+=1
        local_test_ac = (res*eye(output_count)).sum()/res.sum()
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
            for gg in net.groups.values():
                if hasattr(gg,'w'):
                    data['w_%s'%gg.tag] = gg.w.clone().detach()
            pickle.dump(data,handle)       
            
    return net,trn,training_ac,test_ac,loss_hist,res,status