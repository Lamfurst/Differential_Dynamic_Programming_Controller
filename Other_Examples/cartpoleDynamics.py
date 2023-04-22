import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd.numpy import sin, cos



# define dynamics function
def dynamics(state, action):
    # define constants
    x, x_dot, theta, theta_dot = state
    dt = 0.05
    
    # Define constants for the cartpole system
    m_c = 1.0  # mass of the cart
    m_p = 0.1  # mass of the pole
    L = 1.0  # length of the pole
    g = 9.81  # gravitational acceleration
    
    # Compute the force acting on the cart
    F = action
    
    # Compute the sin and cos of the angle
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Compute the derivatives of the state variables
    x_ddot = (F + m_p * L * theta_dot**2 * sin_theta) / (m_c + m_p)
    theta_ddot = (g * sin_theta - cos_theta * (F + m_p * L * theta_dot**2 * sin_theta) / (m_c + m_p)) / L
    
    # Compute the new state variables
    x_new = x + x_dot * dt
    x_dot_new = (x_dot + x_ddot * dt).squeeze()
    theta_new = theta + theta_dot * dt
    theta_dot_new = (theta_dot + theta_ddot * dt).squeeze()
    
    # Return the new state as a NumPy array
    return np.array([x_new, x_dot_new, theta_new, theta_dot_new])


class CartpoleCost:
    def __init__(self, x_final, terminal_scale, Q, R):
        self.x_final = x_final
        self.terminal_scale = terminal_scale
        self.Q = Q
        self.R = R

    def running_cost(self, x, u):
        Q = self.Q
        R = self.R
        dx = self.x_delta(self.x_final, x)
        u = u[np.newaxis]
        return np.squeeze(dx.T @ Q @ dx + u.T @ R @ u)

    def terminal_cost(self, x):
        Q = self.Q
        dx = self.x_delta(self.x_final, x)
        return self.terminal_scale * dx @ Q @ dx
    
    @staticmethod
    def x_delta(x1, x2):
        dx = x1 - x2
        d_theta = np.mod(dx[2] + np.pi, 2 * np.pi) - np.pi
        # d_theta = np.arccos(cos(dx[2])
        return np.array([dx[0], dx[1], d_theta, dx[3]])