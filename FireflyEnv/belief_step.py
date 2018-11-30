import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .env_utils import *
from .env_variables import *

class BeliefStep(nn.Module):
    def __init__(self, dt):
        super(self.__class__, self).__init__()

        self.dt = dt
        self.box = WORLD_SIZE
        return

    def dynamics(self, x, a, theta, w):
        dt = self.dt
        px, py, ang = x
        ang = ang + -theta[1] * a[1] * dt + w[2]
        ang = range_angle(ang)
        px = px + theta[0] * a[0] * torch.cos(ang) * dt + w[0]
        py = py + theta[0] * a[0] * torch.sin(ang) * dt + w[1]
        px = torch.clamp(px, -self.box, self.box)
        py = torch.clamp(py, -self.box, self.box)
        return torch.stack((px, py, ang))

    def A(self, x_, a, theta):
        dt = self.dt
        px, py, ang = x_
        A_ = torch.eye(3)
        A_[0, 2] = - theta[0] * a[0] * torch.sin(ang) * dt
        A_[1, 2] = theta[0] * a[0] * torch.cos(ang) * dt
        return A_

    def L(self, x, a, theta, w):
        dt = self.dt
        x_ = self.dynamics(x, a, theta, w)
        px, py, ang = x_
        L = torch.eye(3, 3)
        L[0, 2] = -theta[0] * a[0] * torch.sin(ang) * dt
        L[1, 2] = theta[0] * a[0] * torch.cos(ang) * dt
        return torch.eye(3) #+ L

    def forward(self, x, P, a, theta):
        #noise = torch.stack([theta[2], theta[2], torch.tensor(-4.)])
        noise = torch.zeros(3)
        noise[0], noise[1] = theta[2], theta[2]
        #noise[2] = theta[2]
        noise[2] = torch.tensor(-4.)
        Q = torch.diag(torch.exp(noise*2))
        I = torch.eye(3)
        w = torch.exp(noise) * torch.randn(3)
        zeros2 = torch.zeros(2)

        x_ = self.dynamics(x, a, theta, w*0)
        A = self.A(x_, a, theta)   # A = self.A(x, a, w*0)
        L = self.L(x, a, theta, w)

        P_ = A.mm(P).mm(A.t()) + L.mm(Q).mm(L.t())
        x = x_
        P = P_
        P = (P + P.t())/2  + 1e-6 * I# make symmetric to avoid computational overflows
        return x, P
