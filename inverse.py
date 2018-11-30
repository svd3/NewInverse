import torch
import torch.nn as nn
import numpy as np
from numpy import pi
from DDPGv2 import Agent, Noise

from FireflyEnv import Model
from FireflyEnv import pos_init
from FireflyEnv.gym_input import true_params

from collections import deque
rewards = deque(maxlen=100)

batch_size = 64
true_params = [p.data.clone() for p in true_params]
env = Model(true_params[-1])

state_dim = env.state_dim
action_dim = env.action_dim
num_steps = int(env.episode_len)

#coords = [pos_init(2.)]
noise = Noise(action_dim, mean=0., std=0.05)
agent = Agent(state_dim, action_dim, hidden_dim=64, tau=0.001)
agent.load('pretrained/ddpg_new/ddpg_model_4.pth.tar')

def trajectory(coord, theta, agent):
    b, _ = env.reset(theta, coord)#.view(1, -1)
    i = 1
    r = 0
    actions = []
    #bstates = [(coords[0], 0)] #(coords, times)
    while True:
        action = agent.select_action(b, noise)
        #action = agent.actor(b).detach() + torch.randn(2)*0.05 #noise.noise()
        next_state, reward, done, info = env(action.view(-1), theta)
        r += reward
        actions.append(action)
        if info['stop']:
            break
            #b = env.reset(coords[i], theta).view(1, -1)
            #i += 1
            #bstates.append((coords[i], t+1))
        b = next_state#.view(1,-1)
    actions = torch.stack(actions)
    return actions, r

def getLoss(theta, coords, actions, agent, batch_size):
    logPr = 0
    totrew = 0
    for i in range(batch_size):
        #id = np.random.randint(1000)
        id = i
        b, _ = env.reset(theta, coords[id])
        #b = b.view(1, -1)
        true_actions = actions[id]
        for a in true_actions:
            action = agent.actor(b)
            #lgPr = -((a - action)**2).sum()/2
            logPr = logPr -((a.view(-1) - action)**2).sum()/(2 * 0.05**2)
            #b, _ = env._single_step(b, a.view(-1), theta)
            b, reward, done, info = env(a.view(-1), theta)
            #b[2:5] = theta
        #
    neglogPr = -logPr
    avg_rew = totrew
    return neglogPr

#res = [[], [], []] #theta, theta0, stderr

theta0 = torch.cat([torch.zeros(2).uniform_(0.75, 1.25), torch.zeros(1).uniform_(-6, -2)])

coords =[]
actions = []
for t in range(200):
    coord = pos_init(1.)
    action_tr, _ = trajectory(coord, theta0, agent)
    coords.append(coord)
    actions.append(action_tr)

loss0 = getLoss(theta0, coords, actions, agent, 200)

#theta = nn.Parameter(torch.cat([torch.zeros(2).uniform_(0.75, 1.25), torch.zeros(1).uniform_(-6, -2)]))
theta = nn.Parameter(theta0.data.clone() + torch.randn(3)*0.1)
losses = deque(maxlen=100)
optT = torch.optim.Adam([theta], lr=1e-3)
for ep in range(1000):
    logPr = 0
    totrew = 0
    for i in range(200):
        #id = np.random.randint(1000)
        id = i
        b, _ = env.reset(theta, coords[id])
        b[2:5] = theta
        #b = b.view(1, -1)
        true_actions = actions[id]
        for a in true_actions:
            action = agent.actor(b)
            #lgPr = -((a - action)**2).sum()/2
            logPr = logPr -((a.view(-1) - action)**2).sum()/(2 * 0.05**2)
            #b, _ = env._single_step(b, a.view(-1), theta.data)
            b, reward, done, info = env(a.view(-1), theta)
            b[2:5] = theta
        #
    neglogPr = -logPr
    avg_rew = totrew
    loss = neglogPr
    losses.append(loss.data)
    print(ep, np.round(loss.data.item(), 6), np.mean(losses), loss0.data)
    optT.zero_grad()
    loss.backward()
    optT.step()

from torch.autograd import grad
loss = getLoss(theta, coords, actions, agent, 200) * (1/0.05)**2
grads = grad(loss, theta, create_graph=True)[0]
H = torch.zeros(3,3)
for i in range(3):
    H[i] = grad(grads[i], theta, retain_graph=True)[0]

I = H.inverse()
stderr = torch.sqrt(I.diag())

res[0].append(theta.data)
res[1].append(theta0.data)
res[2].append(stderr.data)

result = {'theta': torch.stack(res[0]), 'theta0': torch.stack(res[1]), 'stderr': torch.stack(res[2])}
torch.save(result, 'corr_result.pkl')
