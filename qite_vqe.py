import tensorcircuit as tc
import torch
import torch.optim as optim

import numpy as np

from utils import optimality_ratio, expectation, solve_linear_equation, solve_linear_equation_lstsq, norm
 
import json,os

tc.set_backend('pytorch')


# act pauli string on current state
def circuit_operate(num_qubits,operation,current_state):
    circ = tc.Circuit(num_qubits,inputs=current_state)
    for pauli in operation.split('_'):
        if pauli[0] == 'Z':
            circ.z(int(pauli[1:]))
        elif pauli[0] == 'X':
            circ.x(int(pauli[1:]))
        elif pauli[0] == 'Y':
            circ.y(int(pauli[1:]))
    return circ.state()


# compute matrix_s and vector_b
def get_matrix_s(num_qubits,operations,current_state):
    states = torch.stack([circuit_operate(num_qubits,op,current_state) for op in operations], dim=0)
    matrix_s = torch.matmul(states.conj(), states.T)
    matrix_s.fill_diagonal_(1)
    return matrix_s.real

def get_vector_b(num_qubits, current_state, operations, hamil_diag, dt = None):
    operated_states = torch.stack([circuit_operate(num_qubits, op, current_state) for op in operations])
    term1 = torch.sum(operated_states.conj() * hamil_diag * current_state, dim=1)
    if dt is None:
        vector_b = term1
    else: # if dt is not None, use 2-order taylor expansion
        c = norm(current_state, hamil_diag, dt=dt)
        term1 = (c**-0.5) * term1
        term2 = - ((c**-0.5) -1) * torch.sum(operated_states.conj() * current_state, dim=1) / dt
        vector_b = term1 + term2
    return vector_b.imag


def get_matrix_s_vector_b_varQITE(circuit, params, hamil_diag): # hamil_diag: (2**num_qubits)
    jacobian_real = torch.autograd.functional.jacobian(lambda x: torch.real(circuit(x)), params).T
    jacobian_imag = torch.autograd.functional.jacobian(lambda x: torch.imag(circuit(x)), params).T
    # jacobian: \frac{\partial \ket{\psi}_j}{\partial \theta_i}
    jacobian = jacobian_real + 1j*jacobian_imag # shape: (len(params), 2**num_qubits)
    # conj_jacobian: \frac{\partial \bra{\psi}_j}{\partial \theta_i}
    conj_jacobian = jacobian_real - 1j*jacobian_imag # shape: (len(params), 2**num_qubits)
    # ket: \ket{\psi}
    ket = circuit(params) # shape: (2**num_qubits)
    # bra: \bra{\psi}
    bra = ket.conj() # shape: (2**num_qubits)

    # varQITE matrix_s
    # term1： conj_jacobian @ jacobian.T
    term1 = torch.matmul(conj_jacobian, jacobian.T)
    # # term2: conj_jacobian @ |ψ⟩ × ⟨ψ| @ jacobian.T
    # term2 = torch.matmul(torch.matmul(conj_jacobian, torch.outer(ket, bra)), jacobian.T)
    # matrix_s = term1 + term2
    matrix_s = term1

    # varQITE vector_b
    # - conj_jacobian @ H @ |ψ⟩
    vector_b = - torch.matmul(conj_jacobian, hamil_diag * ket)

    return matrix_s.real.detach().clone(), vector_b.real.detach().clone()

    

class QITE_VQE:
    def __init__(self, num_qubits, hamil_diag, operations, initial_state = None, mode = 'AWQV', dt = 0.05, lr = 0.05, lam = 1.1, mu = 0.9, fix_step = 0, constant_c = True, linear_opt_args = None):
        """
        num_qubits: number of qubits
        hamil_diag: Hamiltonian in diagonal form (1-D tensor)
        operations: list of pauli strings, like ['Y0','Z0_Y1','Y2_X3',...]
        initial_state: initial state
        mode: AWQV, VQE, QITE, cQITE, varQITE
        dt: time step of QITE
        lr: learning rate
        lam: hyper-parameter in weighting function
        mu: hyper-parameter in weighting function
        fix_step: number of steps to fix w
        constant_c: whether to use constant normalization scale in computing vector_b
        linear_opt_args: arguments for linear solver using Adam (lr, max_iter, tol)
        """

        # problem properties
        self.num_qubits = num_qubits
        self.hamil_diag = hamil_diag
        self.min_val = hamil_diag.min().item() # minimum eigenvalue of hamil_diag
        self.max_val = hamil_diag.max().item() # maximum eigenvalue of hamil_diag; used for compute approx ratio
        self.min_indices = torch.where(torch.abs(self.hamil_diag-self.min_val)<=1e-6)[0] # indices of minimum eigenvalues; used for compute ground state probability

        # define ansatz
        self.operations = operations

        # set initial state and current state
        if initial_state is None:
            initial_state = tc.Circuit(self.num_qubits).state()
        self.initial_state = initial_state

        self.current_state = self.initial_state

        # hyper-parameters on parameters update
        self.dt = dt # delta tau
        self.lr = lr # learning rate
        self.lam = lam # lambda in update weight of gradient
        self.mu = mu # mu in update weight of gradient
        self.fix_step = fix_step
        self.constant_c = constant_c

        # state preparation method
        self.mode = mode # AWQV or VQE or QITE or cQITE or varQITE

        # set function of yielding circuit
        self.circuit_func = self.set_circuit()
        self.vmap_circuit = self.vmap_circuit()

        # adam linear solver settings
        if linear_opt_args is None:
            linear_opt_args = [0.01,20000,1e-8]
        self.linear_lr, self.max_linear_iter, self.linear_tol = linear_opt_args

        # results
        self.parameters = []
        self.expectations = [expectation(self.current_state, self.hamil_diag).item()]
        self.opt_ratios = [optimality_ratio(self.current_state, self.min_indices).item()]
        self.best_iter = 0
        self.iter = 0
        self.w_history = [0]
        if self.mode == 'VQE': # w of VQE is always 1
            self.w_history = [1]
        self.norm_history = []
        self.res = {}

        # tmp paramters for varQITE
        self.qite_params = None
        def varQITE_circuit(params):
            circ = self.circuit_func(params, self.initial_state)
            return circ.state()
        self.varQITE_circuit = varQITE_circuit
        

    def qite_step(self, mode = None):
        if mode is None:
            mode = self.mode
        if mode == 'varQITE':
            matrix_s, vector_b = get_matrix_s_vector_b_varQITE(self.varQITE_circuit, self.qite_params, self.hamil_diag)
        else:
            if self.constant_c:
                vector_b = get_vector_b(self.num_qubits, self.current_state, self.operations, self.hamil_diag)
            else:
                vector_b = get_vector_b(self.num_qubits, self.current_state, self.operations, self.hamil_diag, self.dt)
            matrix_s = get_matrix_s(self.num_qubits, self.operations, self.current_state)

        if torch.norm(matrix_s - torch.eye(len(matrix_s))) < 1e-6: 
            # if matrix_s is identity matrix, a = b 
            vector_a = vector_b
        else:
            # solve linear equation using lstsq and Adam
            vector_a_lstsq, residual_norm_lstsq = solve_linear_equation_lstsq(matrix_s, vector_b, display = False)
            vector_a_adam, residual_norm_adam = solve_linear_equation(matrix_s, vector_b, None, lr = self.linear_lr, n_iterations = self.max_linear_iter, tol = self.linear_tol, display = False)
            self.norm_history.append(min(residual_norm_lstsq.item(),residual_norm_adam.item())) # record the minimum residual norm of the two solvers at each step
            # choose the solver with smaller residual norm
            if residual_norm_lstsq < residual_norm_adam:
                vector_a = vector_a_lstsq
            else:
                vector_a = vector_a_adam
                
        qite_update_term = - 2 * vector_a

        return qite_update_term


    def update_grad_weight(self):
        if self.mode == 'AWQV': # only AWQV updates w
            if len(self.expectations) <= self.fix_step + 1:
                w = 0
            else:
                expectations_plus = np.array(self.expectations[1:])
                expectations_minus = np.array(self.expectations[:-1])
                delta = - (expectations_plus - expectations_minus)
                delta = delta[self.fix_step:]
                w = self.mu * self.w_history[-1] + (1 - self.mu) * (1 - self.lam * delta[-1] / np.mean(delta))
                w = min(max(w,self.w_history[-1]),1)
            self.w_history.append(w) # record w at each step
            

    def run(self, max_iter = 50, display_interval = 0, initial_params = None, optimizer = None):
        if display_interval != 0:
            print(f"iter 0: exp = {self.expectations[-1]:.4f}, approx = {(self.expectations[-1]-self.max_val)/(self.min_val-self.max_val):.4f}, opt = {self.opt_ratios[-1]:.6f}",flush=True)

        best_exp = self.expectations[-1]
        
        if initial_params is None:
            params = - self.qite_step(mode = "QITE") * self.dt
        else:
            params = initial_params
        params = params.clone().detach().requires_grad_(True)

        # define optimizer
        if optimizer is None:
            optimizer = optim.SGD([params], lr = self.lr)
        elif optimizer == 'Adam':
            optimizer = optim.Adam([params], lr = self.lr)
        elif optimizer == 'LBFGS':
            optimizer = optim.LBFGS([params], lr = self.lr)
        else:
            optimizer = optim.SGD([params], lr = self.lr)


        for i in range(1,max_iter+1):
            if self.mode == 'varQITE':
                self.qite_params = params.clone().detach().requires_grad_(True)
            
            if self.mode == 'QITE':
                state = self.vmap_circuit(params.unsqueeze(0), self.current_state)[0] # QITE apply operations on current state
            else:
                state = self.vmap_circuit(params.unsqueeze(0), self.initial_state)[0] # other models apply operations on initial state

            # compute expectation
            exp = expectation(state, self.hamil_diag)

            # update current state
            self.current_state = state.detach().clone()

            # compute ground state probability
            opt = optimality_ratio(self.current_state, self.min_indices)

            # record expectation and ground state probability
            self.expectations.append(exp.item())
            self.opt_ratios.append(opt.item())

            
            if self.mode == 'QITE': # QITE appends each step's parameters
                self.parameters.append(params.clone().detach().numpy().tolist()) 

            # save the best result
            if exp.item() < best_exp:
                best_exp = exp.item()
                self.best_iter = i # record the best iteration (For QITE, the best parameters are self.parameters[:best_iter])
                if self.mode != 'QITE':
                    self.parameters = params.clone().detach().numpy().tolist() # other models update the best parameters

            # display the loss every display_interval iterations
            if display_interval != 0:
                if i % display_interval == 0:
                    print(f"iter {i}: exp = {self.expectations[-1]:.4f}, approx = {(self.expectations[-1]-self.max_val)/(self.min_val-self.max_val):.4f}, opt = {self.opt_ratios[-1]:.6f} (last_w = {self.w_history[-1]:.4f})",flush=True)
        
            # backpropagation
            if i != max_iter:
                if self.mode == 'QITE':
                    params = - self.qite_step() * self.dt  # QITE updates parameters
                else:
                    # zero grad
                    optimizer.zero_grad()

                    # LBFGS
                    if isinstance(optimizer, torch.optim.LBFGS):
                        def closure():
                            optimizer.zero_grad()
                            state = self.vmap_circuit(params.unsqueeze(0), self.initial_state)[0]
                            exp_closure = expectation(state, self.hamil_diag)
                            exp_closure.backward()
                            return exp_closure

                        if self.w_history[-1] != 1:
                            correction = self.qite_step() # cQITE term
                            self.update_grad_weight() # update w
                            # correct the gradient
                            params.grad = self.w_history[-1] * (torch.norm(correction) / torch.norm(params.grad)) * params.grad + (1 - self.w_history[-1]) * correction * self.dt / self.lr 
                            # params.grad = self.w_history[-1]  * params.grad + (1 - self.w_history[-1]) * (torch.norm(params.grad) / torch.norm(correction)) * correction * self.dt / self.lr

                        optimizer.step(closure)

                    else:
                        exp.backward() # backward propagation
                        
                        if self.w_history[-1] != 1:
                            correction = self.qite_step() # cQITE term
                            self.update_grad_weight() # update w
                            # correct the gradient
                            params.grad = self.w_history[-1] * (torch.norm(correction) / torch.norm(params.grad)) * params.grad + (1 - self.w_history[-1]) * correction * self.dt / self.lr
                            # params.grad = self.w_history[-1]  * params.grad + (1 - self.w_history[-1]) * (torch.norm(params.grad) / torch.norm(correction)) * correction * self.dt / self.lr

                        optimizer.step() # update parameters
                        
        if display_interval != 0:
            print(f"best iter: {self.best_iter}, exp: {self.expectations[self.best_iter]:.4f}, approx: {(self.expectations[self.best_iter]-self.max_val)/(self.min_val-self.max_val):.4f}, opt: {self.opt_ratios[self.best_iter]:.6f}",flush=True)
            
        self.iter = max_iter
        self.set_results()
            
    def set_circuit(self):
        def circuit_func(params, inputs):
            circ = tc.Circuit(self.num_qubits, inputs = inputs)
            for idx, operation in enumerate(self.operations):
                pauli_list = [pauli[0] for pauli in operation.split('_')]
                qubit_list = [int(pauli[1:]) for pauli in operation.split('_')]
                if len(pauli_list) == 1:
                    # single qubit rotations
                    if pauli_list[0] == 'Y':
                        circ.ry(qubit_list[0], theta = params[idx])
                    elif pauli_list[0] == 'Z':
                        circ.rz(qubit_list[0], theta = params[idx])
                    elif pauli_list[0] == 'X':
                        circ.rx(qubit_list[0], theta = params[idx])
                    elif pauli_list[0] == 'M': # X=Y mixer
                        circ.rx(qubit_list[0], theta = params[idx])
                        circ.ry(qubit_list[0], theta = params[idx])
                else:
                    # multi-qubit rotations
                    for q_idx, pauli in enumerate(pauli_list):
                        if pauli == 'Y':
                            circ.rx(qubit_list[q_idx], theta = torch.pi/2)
                        elif pauli == 'X':
                            circ.h(qubit_list[q_idx])
                    for n,m in [(qubit_list[i],qubit_list[i+1]) for i in range(len(qubit_list)-1)]:
                        circ.cnot(n,m)
                    circ.rz(qubit_list[-1], theta = params[idx])
                    for n,m in [(qubit_list[i],qubit_list[i+1]) for i in range(len(qubit_list)-1)][::-1]:
                        circ.cnot(n,m)
                    for q_idx, pauli in enumerate(pauli_list):
                        if pauli == 'Y':
                            circ.rx(qubit_list[q_idx], theta = -torch.pi/2)
                        elif pauli == 'X':
                            circ.h(qubit_list[q_idx])
            return circ
        return circuit_func
    

    def vmap_circuit(self):
        def vmap_circuit_func(params, inputs):
            circ = self.circuit_func(params, inputs)
            return circ.state()
        circuit_vmap = tc.set_backend('pytorch').vmap(vmap_circuit_func, vectorized_argnums=0)
        circuit_batch = tc.interfaces.torch_interface(circuit_vmap, jit=True)
        return circuit_batch

        
    def set_results(self): # set results for saving
        self.res = {
                    # problem info
                    'size': self.num_qubits,
                    'opt_val': self.min_val,
                    # method info
                    'method': self.mode,
                    'operations': self.operations,
                    'dt': self.dt,
                    'lr': self.lr,
                    'lam': self.lam,
                    'mu': self.mu,
                    'fix_step': self.fix_step,
                    'constant_c': self.constant_c,
                    # result
                    'parameters': self.parameters,
                    'opt_ratios': self.opt_ratios,
                    'expectations': self.expectations,
                    'approx_ratios': [(self.expectations[i]-self.max_val)/(self.min_val-self.max_val) for i in range(len(self.expectations))],
                    'iter': self.iter,
                    'best_iter': self.best_iter,
                    'norm_history': self.norm_history,
                    'w_history': self.w_history
                }
        
    def save(self, file_path):
        folder, _ = os.path.split(file_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file_path, 'w') as f:
            json.dump(self.res, f)
        print(f"Results saved to {file_path}",flush=True)
