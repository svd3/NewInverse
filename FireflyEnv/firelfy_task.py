import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .rewards import *
from .terminal import *
from .belief_step import BeliefStep

from .env_utils import *
from .env_variables import *
from .plotter import Render

class Model(nn.Module):
    def __init__(self, log_rew_width):
        super(self.__class__, self).__init__()
        # constants
        self.dt = DELTA_T
        self.action_dim = ACTION_DIM
        self.goal_radius = GOAL_RADIUS
        self.terminal_vel = TERMINAL_VEL
        self.episode_len = EPISODE_LEN
        self.episode_time = EPISODE_LEN * self.dt

        self.Bstep = BeliefStep(self.dt)
        self.reward_param = log_rew_width

        self.rendering = Render()
        self.log = True

        self.reset(None)

    def reset(self, theta, init=None):
        self.time = torch.zeros(1)
        if init is None:
            init = pos_init(self.Bstep.box)
        if theta is None:
            theta = torch.zeros(3)
            #theta[:2] = torch.zeros(2).uniform_(0.75, 1.25)
            theta[0] = torch.zeros(1).uniform_(1.5, 2.5) #velocity gain
            theta[1] = torch.zeros(1).uniform_(1., 2.) #ang. velocity gain
            theta[2] = torch.zeros(1).uniform_(-6, -2)
        r, ang, rel_ang = init
        pos = torch.cat([r*torch.cos(ang), r*torch.sin(ang)])
        ang = ang + pi + rel_ang
        ang = range_angle(ang)
        self.x = torch.cat([pos, ang])
        relx = torch.cat([r, rel_ang])

        # covariance
        self.P = torch.eye(3) * 1e-8
        vecL = vectorLowerCholesky(self.P)

        state = torch.cat([relx, theta, self.time, vecL])
        self.state_dim = state.size(0)
        return state, theta

    def forward(self, a, theta):
        x, P, time = self.x, self.P, self.time
        time += self.dt
        x, P = self.Bstep(x, P, a, theta)

        pos, ang = x[:2], x[2]
        r = torch.norm(pos).view(-1)
        rel_ang = ang - torch.atan2(-pos[1], -pos[0]).view(-1)
        relx = torch.cat([r, rel_ang])
        vecL = vectorLowerCholesky(P)
        state = torch.cat([relx, theta, time, vecL])

        terminal = self._isTerminal(x, a)
        done = time >= self.episode_time
        #reward = terminal * self._get_reward(x, P, time, a) - 1
        reward = -1 *torch.ones(1) #- 0.1 * vels.norm()**2
        self.x, self.P, self.time = x, P, time
        if terminal:
            reward = reward + self._get_reward(x, P, time, a)
            state, _ = self.reset(theta=theta)
        return state, reward.view(-1), done, {'stop': terminal}

    def _get_reward(self, x, P, time, a):
        rew_param = torch.exp(2 * self.reward_param)
        reward = rewardFunc(rew_param, x, P, time, a, scale=80)
        return reward

    def _isTerminal(self, x, a, log=True):
        goal_radius = self.goal_radius
        terminal_vel = self.terminal_vel
        #terminal, reached_target = is_terminal_velocity(x, a, goal_radius, terminal_vel)
        terminal, reached_target = is_terminal_action(x, a, goal_radius, terminal_vel)
        if terminal and log:
            #print("Stopped. {:0.2f}".format(torch.norm(x[:2]).item()))
            pass
        if reached_target and self.log:
            #pass
            print("Goal!!")
        return terminal.item() == 1

    def _single_step(self, state, a, theta):
        relx = state[:2]
        r, rel_ang = state[0], state[1]
        time = state[5].view(-1)
        #ang = torch.zeros(1).uniform_(-pi, pi)
        ang = torch.zeros(1)
        pos = torch.cat([r*torch.cos(ang), r*torch.sin(ang)])
        ang = ang + pi + rel_ang
        ang = range_angle(ang)
        x = torch.cat([pos, ang])

        vecL = state[6:]
        P = inverseCholesky(vecL)

        time = time  + self.dt
        x, P = self.Bstep(x, P, a, theta)

        pos, ang = x[:2], x[2]
        r = torch.norm(pos).view(-1)
        rel_ang = ang - torch.atan2(-pos[1], -pos[0]).view(-1)
        relx = torch.cat([r, rel_ang])
        vecL = vectorLowerCholesky(P)
        #new_state = torch.zeros(self.state_dim)
        new_state = torch.cat([relx, theta, time, vecL])

        terminal = self._isTerminal(x, a)
        #done = time >= self.episode_time
        reward = -1 *torch.ones(1) #- 0.1 * vels.norm()**2
        if terminal:
            reward = reward + self._get_reward(x, P, time, a)
        #    state = self.reset(theta=g)

        return new_state, reward

    def render(self):
        goal = torch.zeros(2)
        self.rendering.render(goal, self.x, self.P)
