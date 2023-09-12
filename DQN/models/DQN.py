from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import random
from collections import deque

class net(nn.Module):
    def __init__(self,channel,num_action):
        super(net,self).__init__()
        self.conv1=nn.Conv2d(channel,16,kernel_size=8,stride=4,dtype=torch.float32) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2,dtype=torch.float32)  
        self.in_features=32*9*9
        self.fc1=nn.Linear(self.in_features,256,dtype=torch.float32)
        self.fc2=nn.Linear(256,num_action,dtype=torch.float32)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x
class dqn(nn.Module):
    def __init__(self,channel, num_action, buffer_size ,*args,**kwrgs):
        super(dqn,self).__init__()
        self.num_action = num_action
        self.Q = net(channel,num_action)
        self.target_Q = net(channel,num_action)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.replay_buffer = deque(maxlen = int(buffer_size))
    def sample(self, batch_size):
        batch_size = min(batch_size,len(self.replay_buffer))
        s,ns,a,r,done = (torch.Tensor() for i in range(5))
        for i in random.sample(self.replay_buffer,batch_size):
            observations = map(lambda x:torch.tensor(x).unsqueeze(dim=0),i)
            s,ns,a,r,done = map(torch.cat,zip([s,ns,a,r,done],observations))
        return (s,ns),(a,r,done)
    def forward(self,x):
        return self.Q(x)
    def select_action(self,x , eps = 1.0):
        if self.training and (np.random.rand() < eps):
            a = np.random.randint(0, self.num_action)
            with torch.no_grad():
                x = torch.from_numpy(x).unsqueeze(0).to(dtype = torch.float32,device=self.device())
                q = self.forward(x).squeeze()[a].item()
            return a, q
        else:
            with torch.no_grad():
                x = torch.from_numpy(x).unsqueeze(0).to(dtype = torch.float32,device=self.device())
                q = self.forward(x)
                a = torch.max(q,dim=1,keepdim=True)
            return a[1].item(), a[0].item()
    def get_error(self, batch_size, gamma):
        states,observations=self.sample(batch_size)
        s, ns = map(lambda x: x.to(dtype = torch.float32,device = self.device()), states)
        a, r, done = map(lambda x:x.to(dtype = torch.float32,device = self.device()).unsqueeze(dim=1),observations)
        q_value = self.forward(s).gather(1,a.long())
        with torch.no_grad():
            ns_q_value=self.target_Q(ns).max(1,keepdim=True)[0]
        td_target=r+(1.-done)*gamma*ns_q_value
        return q_value ,td_target 
    def target_update(self,soft :bool = True,tau : float = 0.1) :
        if soft:
            q_params = self.Q.state_dict()
            target_Q = self.target_Q.state_dict()
            for key in target_Q:
                target_Q[key] = q_params[key]*tau + target_Q[key]*(1-tau)
            self.target_Q.load_state_dict(target_Q)
        else :
            self.target_Q.load_state_dict(self.Q.state_dict())
    def device(self):
        return next(self.parameters()).device