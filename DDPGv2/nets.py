import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        num_outputs = action_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, num_outputs)

    def forward(self, inputs):
        x = torch.relu(self.linear1(inputs))
        x = torch.relu(self.linear2(x))
        mu = torch.tanh(self.mu(x))
        #mu = F.sigmoid(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear_action = nn.Linear(action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

        x = np.arange(-1, 1.005, 0.05)
        x,y = np.meshgrid(x, x)
        grid = np.vstack([x.ravel(), y.ravel()])
        self.actions = torch.Tensor(grid).t()
        self.nactions = self.actions.size(0)

    def forward(self, inputs, actions):
        x = torch.relu(self.linear1(inputs))
        a = torch.relu(self.linear_action(actions))
        x = torch.cat((x, a), dim=1)
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))

        V = self.V(x)
        return V

    def optimalValue(self, inputs):
        batch_size, in_dim = inputs.size()
        x = inputs.unsqueeze(1).repeat(1, self.nactions, 1).view(-1, in_dim)
        a = self.actions.repeat(batch_size)
        v = self(x, a)
        vs = v.split(self.nactions)
        return torch.stack(list(map(torch.max, vs)))
