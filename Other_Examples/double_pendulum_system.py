import torch
import numpy as np
from mppi import MPPI

def dynamics_analytic(state, control):
    # state = [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    # control = [u]
    # state_dot = [x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]
    # state_dot = f(state, control)
    # batch input, state shape = (batch_size, 6)
    # batch input, control shape = (batch_size, 1)
    next_state = None
    dt = 0.05
    g = 9.81
    mc = 1
    mp1 = 0.1
    mp2 = 0.1
    L1 = 0.5
    L2 = 0.5

    # --- Start dynamic model
    
    if len(state.shape) == 1:
        state = state.unsqueeze(0)
        control = control.unsqueeze(0)
    batch_size = state.shape[0]
    next_state_list = torch.zeros(batch_size, 6)
    for i in range(batch_size):
        theta = torch.tensor([state[i,0], state[i,2], state[i,4]], dtype=torch.float32)
        theta_dot = torch.tensor([state[i,1], state[i,3], state[i,5]], dtype=torch.float32)
        theta_ddot = torch.zeros(3,dtype=torch.float32)
        D = torch.zeros(3,3,dtype=torch.float32)
        D[0,0] = mc+mp1+mp2
        D[0,1] = (1/2*mp1+mp2)*L1*torch.cos(theta[1])
        D[0,2] = (1/2*mp2)*L2*torch.cos(theta[2])
        D[1,0] = (1/2*mp1+mp2)*L1*torch.cos(theta[1])
        D[1,1] = (1/3*mp1+mp2)*L1**2
        D[1,2] = (1/2*mp2)*L1*L2*torch.cos(theta[1]-theta[2])
        D[2,0] = (1/2*mp2)*L2*torch.cos(theta[2])
        D[2,1] = (1/2*mp2)*L1*L2*torch.cos(theta[1]-theta[2])
        D[2,2] = (1/3*mp2)*L2**2
        C = torch.zeros(3,3,dtype=torch.float32)
        C[0,1] = -(1/2*mp1+mp2)*L1*theta_dot[1]*torch.sin(theta[1])
        C[0,2] = -(1/2*mp2)*L2*theta_dot[2]*torch.sin(theta[2])
        C[1,2] = (1/2*mp2)*L1*L2*theta_dot[2]*torch.sin(theta[1]-theta[2])
        C[2,1] = -(1/2*mp2)*L1*L2*theta_dot[1]*torch.sin(theta[1]-theta[2])
        G = torch.zeros(3,dtype=torch.float32)
        G[1] = -1/2*(mp1+mp2)*g*L1*torch.sin(theta[1])
        G[2] = -1/2*mp2*g*L2*torch.sin(theta[2])
        H = torch.zeros(3,dtype=torch.float32)
        H[0] = 1.
        theta_ddot = -torch.matmul(torch.pinverse(D).float(), torch.matmul(C, theta_dot).float() + G - H*control[i,:].float())
        # --- End dynamic model
        theta_dot = theta_dot + theta_ddot*dt
        theta = theta + theta_dot*dt
        next_state = torch.tensor([theta[0], theta_dot[0], theta[1], theta_dot[1], theta[2], theta_dot[2]])
        next_state_list[i,:] = next_state


    
    # print(next_state_list)
    return next_state_list
    

def linearize_pytorch(state,control):
    """
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (6,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (6, 6) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (6, 1) representing Jacobian df/du for dynamics f

    """
    dt = 0.05
    state.requires_grad = True
    control.requires_grad = True

    A, B = torch.autograd.functional.jacobian(dynamics_analytic, (state, control))
    next_state = state+ (A@state + B@control)*dt

    return next_state

# def cost_function(state, control, target_state):
#     """
#         Cost function for the double pendulum
#     """
#     state_cost = torch.sum((state-target_state)**2, dim=-1)
#     # control_cost = torch.sum(control**2, dim=-1)
#     cost = state_cost #+ 0.05*control_cost
#     # print(cost)
#     return cost


class DoublePendulumControl(object):
    """
        Control class for the double pendulum
        All the data should be in tensor form
    """
    # def __init__(self, dynamics,cost_function, num_samples = 100, horizon = 10):
    def __init__(self, dynamics, num_samples = 100, horizon = 10):
        self.dynamics = dynamics
        # self.cost_function = cost_function
        self.target_state = torch.tensor([0,0,0,0,0,0])
        self.state_dim = 6
        self.control_dim = 1
        self.num_samples = num_samples
        self.horizon = horizon
        self.noise_sigma = 80. * torch.eye(self.control_dim)
        self.lambda_value = 0.1
        self.mppi = MPPI(self._compute_dynamics,
                         self._compute_costs,
                         nx = self.state_dim,
                         num_samples=self.num_samples,
                         horizon=self.horizon,
                         noise_sigma=self.noise_sigma,
                         lambda_=self.lambda_value)
        self.cost = 0.

        
    def control_calcu(self, state):
        action = self.mppi.command(state)
        return action

    def _compute_dynamics(self,state,control):
        return self.dynamics(state,control)
    
    def _compute_costs(self,state,control):
        diff = state - self.target_state
        Q = torch.diag(torch.tensor([50., 200., 1000., 130., 1000., 130.]))
        R = torch.tensor([[0.0]])
        state_cost = (diff @ Q @ diff.T).diag()
        control_cost = (control @ R @ control.T).diag()
        cost = state_cost + control_cost
        self.cost = cost
        return cost

    
    