import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd.numpy import sin, cos




# Programmer: Junkai Zhang, April 8th 2023
# Purpose: This program implements the Differential Dynamic Programming (DDP) 
#          algorithm based on the autograd package.
    

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


        miu_1 = 0.
        miu_2 = 0.
        for iter in range(self.max_iter):

            break_flag = False

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
        
    


    