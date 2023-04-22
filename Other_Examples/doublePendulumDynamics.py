
import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd.numpy import sin, cos

def dp_state_def(x1, x2):
    dx = x1 - x2
    d_theta_1 = np.mod(dx[2] + np.pi, 2 * np.pi) - np.pi
    d_theta_2 = np.mod(dx[4] + np.pi, 2 * np.pi) - np.pi

    return np.array([dx[0], dx[1], d_theta_1, dx[3], d_theta_2, dx[5]])

def dynamics_numpy(state,control):
    # state = [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    # control = [u]
    # state_dot = [x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]
    # state_dot = f(state, control)
    # state shape = (6,)
    # control shape = (1, )
    # All the data should be in numpy array
    next_state = None
    dt = 0.05
    g = 9.81
    mc = 1
    mp1 = 0.1
    mp2 = 0.1
    L1 = 0.5
    L2 = 0.5

    # --- Start dynamic model
    theta = np.array([state[0], state[2], state[4]])
    theta_dot = np.array([state[1], state[3], state[5]])
    theta_ddot = np.zeros(3)
    D = np.array([[mc+mp1+mp2, (1/2*mp1+mp2)*L1*np.cos(theta[1]), (1/2*mp2)*L2*np.cos(theta[2])],
                  [(1/2*mp1+mp2)*L1*np.cos(theta[1]), (1/3*mp1+mp2)*L1**2, (1/2*mp2)*L1*L2*np.cos(theta[1]-theta[2])],
                  [(1/2*mp2)*L2*np.cos(theta[2]), (1/2*mp2)*L1*L2*np.cos(theta[1]-theta[2]), (1/3*mp2)*L2**2]])
    C = np.array([[0, -(1/2*mp1+mp2)*L1*theta_dot[1]*np.sin(theta[1]), -(1/2*mp2)*L2*theta_dot[2]*np.sin(theta[2])],
                  [0, 0, (1/2*mp2)*L1*L2*theta_dot[2]*np.sin(theta[1]-theta[2])],
                  [0, -(1/2*mp2)*L1*L2*theta_dot[1]*np.sin(theta[1]-theta[2]), 0]])
    G = np.array([0, -1/2*(mp1+mp2)*g*L1*np.sin(theta[1]), -1/2*mp2*g*L2*np.sin(theta[2])])
    H = np.array([1., 0., 0.])
    # if (np.linalg.det(D) < 1e-6):
    #     D = D + 1e-6*np.eye(D.shape[0])
    theta_ddot = np.linalg.inv(D) @ (H*control - C @ theta_dot - G)
    theta_dot = theta_dot + theta_ddot*dt
    theta = theta + theta_dot*dt
    next_state = np.array([theta[0], theta_dot[0], theta[1], theta_dot[1], theta[2], theta_dot[2]])

    return next_state


class Cost:
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
        d_theta_1 = np.mod(dx[2] + np.pi, 2 * np.pi) - np.pi
        d_theta_2 = np.mod(dx[4] + np.pi, 2 * np.pi) - np.pi

        return np.array([dx[0], dx[1], d_theta_1, dx[3], d_theta_2, dx[5]])