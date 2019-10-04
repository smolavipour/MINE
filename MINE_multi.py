import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd



class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output
    
def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.2):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch
    #joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    joint = torch.autograd.Variable(torch.FloatTensor(joint)) #Sina
    #marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)) #Sina
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    
    # unbiasing use moving average
    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
    # use biased estimator
#     loss = - mi_lb
    
    mine_net_optim.zero_grad()
    #autograd.backward(loss)
    loss.backward() #Sina
    mine_net_optim.step()
    return mi_lb, ma_et

def learn_mine_biased(batch, mine_net, mine_net_optim):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch
    #joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    joint = torch.autograd.Variable(torch.FloatTensor(joint)) #Sina
    #marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)) #Sina
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    
    
    # use biased estimator
    loss = - mi_lb
    
    mine_net_optim.zero_grad()
    #autograd.backward(loss)
    loss.backward() #Sina
    mine_net_optim.step()
    return mi_lb


def sample_batch(data, batch_size=300, sample_mode='joint'):
    #Data=(X,Y) where X and Y have a dimension d
    (X,Y)=data    
    N=X.shape[0]
    if sample_mode == 'joint':
        index = np.random.choice(range(N), size=batch_size, replace=False)
        batch = np.concatenate([X[index],Y[index]],axis=1)
    else:
        index_1 = np.random.choice(range(N), size=batch_size, replace=False)
        index_2 = np.random.choice(range(N), size=batch_size, replace=False)
        batch = np.concatenate([X[index_1], Y[index_2]],axis=1)
    return batch


def train(data, mine_net,mine_net_optim, batch_size=300, iter_num=int(5e+3), log_freq=int(1e+3), Unbiased=True):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data,batch_size=batch_size)\
        , sample_batch(data,batch_size=batch_size,sample_mode='marginal')        
        if Unbiased:            
            #Unbiased
            mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        else:            
            #Biased
            mi_lb= learn_mine_biased(batch, mine_net, mine_net_optim)
            
        result.append(mi_lb.detach().cpu().numpy())
        #if (i+1)%(log_freq)==0:
            #print(result[-1])
    return result

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]




##-----------------------------------------------------------------------------------------

n=1000
d=5
True_MI=[]
Estimated_MI=[]
Estimated_MI_detail=[]
Lr=1e-4
display('d')
for rho in np.arange(-0.95,0.99,0.05):
    print(rho)
    Sigma=np.tensordot([[1,rho],[rho,1]],np.eye(d),0)
    Sigma=np.reshape(Sigma,(2,2*d,d),2*d)
    Sigma=np.reshape(Sigma,(2*d,2*d),0)
    #print(Sigma)
    xy = np.random.multivariate_normal( mean=[0]*2*d,
                                      cov=Sigma,
                                     size = n)
    x=xy[0:n,0:d]
    y=xy[0:n,d:2*d]
    data=(x,y)

    # plt indep Gaussian
    #sns.scatterplot(x=x[:,0],y=x[:,1])

    # plt cor Gaussian
    #sns.scatterplot(x=x[:,0],y=y[:,0])

    # plt ind Gaussian
    #sns.scatterplot(x=x[:,0],y=y[:,1])

    mine_net= Mine(input_size=2*d)
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=Lr)

    
    result = train(data,mine_net,mine_net_optim,batch_size=300)
    Estimated_MI_detail.append(result[-200:-1])
    print(-0.5*d*np.log(1-rho**2),sum(np.asarray(result[-100:-1]))/100)
    Estimated_MI.append( sum(np.asarray(result[-100:-1]))/100 ) 
    True_MI.append(-0.5*d*np.log(1-rho**2))
    

plt.plot(np.arange(-0.95,0.99,0.05),np.asarray(True_MI),color='blue')+ plt.plot(np.arange(-0.95,0.99,0.05),np.asarray(Estimated_MI),color='orange')
plt.show()

# open a file, where you ant to store the data
file = open('Mine_multi_5', 'wb')
# dump information to that file
pickle.dump((True_MI,Estimated_MI,Estimated_MI_detail), file)

file.close()