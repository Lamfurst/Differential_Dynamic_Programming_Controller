import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd.numpy import sin, cos
from tqdm import tqdm

# Programmer: Junkai Zhang & Hao Liu,  April 8th 2023
# Purpose: This program is used for demo purpose of ROB498. This program
#          implements the Differential Dynamic Programming (DDP) algorithm based 
#          on the autograd package and achieve the swing-up control of a double
#          pendulum.

def main():
    # Define the dynamics of the double pendulum
    state_dim = 6
    action_dim = 1
    x_final = np.array([.0, .0, .0, .0, .0, .0])

    # Define cost parameters and matrix used in the cost function
    Q = np.diag([50., 200., 1000, 130., 1000, 130]) # Best
    R = np.array([[.3]])
    terminal_scale = 10.0
    cost = Cost(x_final, terminal_scale, Q, R)
    DDP_dynamic = dynamics_numpy

    # Define the controller
    controller = DDPcontroller(DDP_dynamic, cost, T = 10, max_iter= 100)

    # Define the initial state
    initial_state = np.array([0,0,np.pi,0,np.pi,0])
    state = initial_state

    # Define the target state
    target = x_final
    num_steps = 100

    # Define the plotting
    pbar = tqdm(range(num_steps))
    fig, ax = plt.subplots()

    for i in pbar:

        # --- Start plotting
        plt.clf()
        ax = plt.axes(xlim=(state[0]-10, state[0]+10), ylim=(-2, 2))
        ax.set_aspect('equal')
        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Double Pendulum at t={:.2f}'.format(i*0.05))
        x = state[0]
        theta1 = state[2]
        theta2 = state[4]
        L1 = 0.5
        L2 = 0.5
        x1 = x + L1*np.sin(theta1)
        y1 = L1*np.cos(theta1)
        x2 = x1 + L2*np.sin(theta2)
        y2 = y1 + L2*np.cos(theta2)
        plt.plot([x,x1],[0,y1],color='green')   
        plt.plot([x1,x2],[y1,y2],color='black')
        plt.plot(x,0,'o',color='red',markersize=5)
        plt.plot(x1,y1,'o',color='blue',markersize=1)
        # add legend to the two points
        plt.legend(['Link 1','Link 2','Cart','Pendulum'])

        plt.draw()
        plt.pause(1e-17)
        # --- End plotting

        # Use controller to get the action
        action = controller.command(state)
        
        # Use the action to get the next state
        state = DDP_dynamic(state, action)
        state = state.squeeze()

        # Calculate the error between the current state and the target state
        error_i = dp_state_def(state, target)
        error_i = np.linalg.norm(error_i @ np.diag([0.0, 0.0, 1, 0.1, 1, 0.1]))
        pbar.set_description(f'Goal Error: {error_i:.4f}')

        # Check if the goal is reached
        if error_i < 0.1  and i > 50:
            num_steps = i
            print('\n Goal reached')
            break

    plt.show()
    plt.close()

# ------ Define all the auxiliary functions and classes used in the demo ------#

# Define the difference between two states of the double pendulum
def dp_state_def(x1, x2):
    dx = x1 - x2
    d_theta_1 = np.mod(dx[2] + np.pi, 2 * np.pi) - np.pi
    d_theta_2 = np.mod(dx[4] + np.pi, 2 * np.pi) - np.pi

    return np.array([dx[0], dx[1], d_theta_1, dx[3], d_theta_2, dx[5]])

# DoublePendulum Dynamics definition
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

# Define the cost class for the double pendulum
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
    
    # Define DDP controller class
class DDPcontroller:
    # Finite horizon Discrete-time Differential Dynamic Programming(DDP)
    def __init__(
        self, 
        dynamics,
        cost,
        tolerance = 1e-4,
        max_iter = 50,
        T = 100,
        state_dim = 6,
        control_dim = 1,
        rho = 0.9,
        max_dc_iter = 10,
        dt = 0.05
    ):
        self.dynamics = dynamics
        self.cost = cost
        self.running_cost = cost.running_cost
        self.terminal_cost = cost.terminal_cost
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.T = T # Total time steps, actions should be T - 1 times
        self.rho = rho # Backtracking line search parameter
        self.dt = dt

        # Define functions and derivatives needed
        # Define V, Vx, Vxx functions
        self.V = lambda x: self.terminal_cost(x)
        self.Vx = grad(self.V)
        self.Vxx = jacobian(self.Vx)

        # Define L, Lx, Lxx, Lu, Luu, Lxu functions
        self.L = lambda x, u: self.running_cost(x, u)
        self.Lx = grad(self.L, 0)
        self.Lu = grad(self.L, 1)
        self.Lxx = jacobian(self.Lx, 0)
        self.Luu = jacobian(self.Lu, 1)
        self.Lxu = jacobian(self.Lx, 1)
        self.Lux = jacobian(self.Lu, 0)

        # define F, Fx, Fu functions
        self.F = self.dynamics
        
        self.Fx = jacobian(self.F, 0)
        self.Fu = jacobian(self.F, 1)
        self.max_dc_iter = max_dc_iter


    def command(self, state, action = None):
        # DDP algorithm
        # state: current state with shape (state_dim, )
        # return: control action with shape (control_dim, )
        V = self.V
        Vx = self.Vx
        Vxx = self.Vxx
        L = self.L
        Lx = self.Lx
        Lu = self.Lu
        Lxx = self.Lxx
        Luu = self.Luu
        Lxu = self.Lxu
        Lux = self.Lux
        F = self.F
        Fx = self.Fx
        Fu = self.Fu

        # Initialize the state
        x_init = state
        # Initialize the control with random values
        if action is None:
            # U = np.random.rand(self.T - 1, self.control_dim)
            U = np.random.uniform(-1, 1, (self.T - 1, self.control_dim))
        else:
            U = np.repeat(action, self.T - 1, axis = 0).reshape(self.T - 1, self.control_dim)
            
        # Initialize the trajectory
        # TODO: Use rollout function to initialize the trajectory
        X = self._rollout(x_init, U)

        # Initialize the cost
        prev_cost = self._compute_total_cost(X, U)

        # i = 0
        miu_1 = 0.
        miu_2 = 0.
        # while i < self.max_iter:
        for iter in range(self.max_iter):

            break_flag = False

            # i += 1
            # print(str(i) + "i")

            # Backward pass
            # Initialize the cost-to-go
            Vx_val = Vx(X[-1])
            Vxx_val = Vxx(X[-1])
            
            k_list = []
            K_list = []
            
            for t in range(self.T - 2, -1, -1):
                # Compute the cost-to-go
                # TODO: Add miu_1 and miu_2 adjustment
                x = X[t]
                u = U[t]
                Fx_val = Fx(x, u)
                Fu_val = Fu(x, u)
                Lx_val = Lx(x, u)
                Lu_val = Lu(x, u)
                Lxx_val = Lxx(x, u)
                Luu_val = Luu(x, u)
                Lux_val = Lux(x, u)

                # No need ------------------------------------------------------
                # Qx = Lx(X[t], U[t]) + Fx(X[t], U[t]).T @ Vx_val
                # Qu = Lu(X[t], U[t]) + Fu(X[t], U[t]).T @ Vx_val

                # miu_1 = 0
                # miu_2 = 0
                # eye_x = np.eye(self.state_dim)
                # eye_u = np.eye(self.control_dim)
                # Qxx = Lxx(X[t], U[t]) + Fx(X[t], U[t]).T @ (Vxx_val + miu_1 * eye_x) @ Fx(X[t], U[t])
                # # Qux = Lxu(X[i], U[i]) + Fu(X[i], U[i]).T @ Vxx_val @ Fx(X[i], U[i])
                # Qux = Lux(X[t], U[t]) + Fu(X[t], U[t]).T @ (Vxx_val + miu_1 * eye_x) @ Fx(X[t], U[t])
                # Quu = Luu(X[t], U[t]) + Fu(X[t], U[t]).T @ (Vxx_val + miu_1 * eye_x) @ Fu(X[t], U[t]) + miu_2 * eye_u
                # No need ------------------------------------------------------

                Qx = Lx_val + Fx_val.T @ Vx_val
                Qu = Lu_val + Fu_val.T @ Vx_val

                eye_x = np.eye(self.state_dim)
                eye_u = np.eye(self.control_dim)
                Qxx = Lxx_val + Fx_val.T @ (Vxx_val + miu_1 * eye_x) @ Fx_val
                # Qux = Lxu(X[i], U[i]) + Fu(X[i], U[i]).T @ Vxx_val @ Fx(X[i], U[i])
                Qux = Lux_val + Fu_val.T @ (Vxx_val + miu_1 * eye_x) @ Fx_val
                Quu = Luu_val + Fu_val.T @ (Vxx_val + miu_1 * eye_x) @ Fu_val + miu_2 * eye_u
                # Determine whether the Quu is invertible
                det = np.linalg.det(Quu)
                if abs(det) < 1e-6:
                    print("The array is singular and not invertible.")
                    break_flag = True
                    miu_1 += 0.1
                    miu_2 += 0.1
                    break
                
                # Compute the control gain
                k = -np.linalg.inv(Quu) @ Qu
                K = -np.linalg.inv(Quu) @ Qux

                # Add k and K to list
                k_list.append(k)
                K_list.append(K)

                # Update the cost-to-go
                Vx_val = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
                Vxx_val = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            
            if break_flag:
                    continue

            # Reverse the list
            k_list.reverse()
            K_list.reverse()

            # Forward pass
            eps = 1.
            dc_iter = 0
            while dc_iter < self.max_dc_iter:
                dc_iter += 1
                # Compute the trajectory
                X_new = np.zeros_like(X)
                U_new = np.zeros_like(U)
                X_new[0] = X[0].copy()
                for t in range(self.T - 1):
                    delta_x = X_new[t] - X[t]
                    U_new[t] = U[t] + eps *  k_list[t] + K_list[t] @ delta_x
                    X_new[t + 1] = self._compute_dynamics(X_new[t], U_new[t])
                
                # Compute the cost
                cost = self._compute_total_cost(X_new, U_new)
                if (cost < prev_cost):
                    X = X_new
                    U = U_new
                    break
                else:
                    eps *= self.rho

            if abs(prev_cost - cost) < self.tolerance:
                break
            prev_cost = cost

        return U[0]
                
    def _compute_dynamics(self,state,control):
        return self.dynamics(state,control)
    
    def _compute_total_cost(self,state,control):
        # Compute the total cost of the trajectory
        # state: current state
        # control: current control
        # return: total cost
        total_cost = 0.
        for i in range(self.T - 1):
            total_cost += self.cost.running_cost(state[i], control[i])
        total_cost += self.terminal_cost(state[-1])
        return total_cost
    
    def _rollout(self, init_state, controls):
        # Rollout the trajectory
        # state: current state
        # control: current control
        # return: trajectory
        states = [init_state]
        for i in range(controls.shape[0]):

            states.append(self._compute_dynamics(states[-1], controls[i]))
        
        return np.array(states)

    
if __name__ == '__main__':
    print("The simulation takes no more than 70 seconds.")
    print("The first several steps will take longer time than later steps. Please wait patiently.")
    print("In rare cases, the simulation may not converge due to the random action initialization. Please run the code again.")
    main()