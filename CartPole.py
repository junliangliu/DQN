
# coding: utf-8

# In[1]:

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


# In[2]:

#Q(S,a)
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.dense1 = nn.Linear(4, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
    
model = DQN()
# a = Tensor([[1,2,3,4]])
# print(a)
# a = Variable(a)
# a = model(a)
# print(a)
# print(a.max(1)[1])


# In[3]:

#定义回放缓存
memory = deque(maxlen=10000)


# In[4]:

#全局参数设置
BATCH_SIZE = 32
GAMMA = 0.9
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss() #采用最小均方误差是线性回归


# In[5]:

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01

epsilon = INITIAL_EPSILON

def select_action(state):
    
    global epsilon
    sample = random.random()
    
    #epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
    
    if random.random() <= epsilon:
        return random.randint(0,1)
    else:
        s = Variable(FloatTensor(state.reshape(1,4)),volatile=True)
        out = model(s).max(1)[1].data[0]
        return out

a = select_action(np.ones(4))
print(a)


# In[6]:

#优化
def optimize_model(memory, model, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory,BATCH_SIZE)
    [states, actions, rewards, next_states, dones] = zip(*batch)
    
    state_batch = Variable(Tensor(states))
    action_batch = Variable(LongTensor(actions))
    reward_batch = Variable(Tensor(rewards))
    next_states_batch = Variable(Tensor(next_states))
    
    #反向传播时更新参数
    state_action_values = model(state_batch).gather(1, action_batch.view(-1,1))
    
    #仅前向计算，不反向传播
    next_states_batch.volatile = True
    next_state_values = model(next_states_batch).max(1)[0]
    for i in range(BATCH_SIZE):
        if dones[i]:
            next_state_values.data[i]=0
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values.volatile = False
    
    loss = criterion(state_action_values, expected_state_action_values)    

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return 1


# In[ ]:

#尝试次数
for episode in range(10000):  
    
    observation = env.reset()  
    state = observation
    
    #尝试玩一次游戏并优化动作估值函数
    for t in count():  
        env.render()  
        
        action = select_action(state) #选择动作
        observation,reward,done,info=env.step(action)
        next_state = observation
        
        # Store the transition in memory
        memory.append([state, action, reward, next_state, done])
        
        #下一轮迭代
        state = next_state
        
        #优化Q(S,a)
        if len(memory) >= BATCH_SIZE:
            optimize_model(memory, model, optimizer)
            
        if done:              
            break  
        
    #测试动作估值函数
    if episode % 100 == 0:
        
        test_cnt = 10
        total_reward = 0
        for i in range(test_cnt):
            test_state = env.reset()
            for j in count():
                env.render()
                test_action = model(Variable(FloatTensor(test_state.reshape(1,4)),volatile=True)).max(1)[1].data[0]
                test_state, test_reward, test_done, _ = env.step(test_action)
                total_reward += test_reward
                if test_done:
                    break
        ave_reward = total_reward/test_cnt
        print('episode: ',episode,'Evaluation Average Reward:',ave_reward)
        if ave_reward >= 300:
            break


# In[ ]:

#对已训练模型测试表现
test_cnt = 30
for i in range(test_cnt):
    total_reward = 0
    test_state = env.reset()
    for j in count():
        env.render()
        test_action = model(Variable(FloatTensor(test_state.reshape(1,4)),volatile=True)).max(1)[1].data[0]
        test_state, test_reward, test_done, _ = env.step(test_action)
        total_reward += test_reward
        if test_done:
            break
    print('test: ',i,'Test Reward:',total_reward)


# In[ ]:



