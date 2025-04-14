import tensorcircuit as tc
import tensorcircuit.quantum as qu
import torch
import torch.optim as optim
import numpy as np

import networkx as nx

tc.set_backend('pytorch')

def maxcut_hamil_diag(edges, tf = False):
    m = max(max(edge[:2]) for edge in edges) + 1
    ls = []
    weight = []
    for edge in edges:
        if len(edge) == 2:
            i,j = edge
            weight.append(1)
        else:
            i,j,w = edge
            weight.append(w)
        l = [0] * m
        l[i] = 3 # I:0, X:1, Y:2, Z:3
        l[j] = 3
        ls.append(l)
    if tf:
        H = (np.sum(weight) - qu.PauliStringSum2COO_tf(ls,weight).values) / 2
    else:
        H = (np.sum(weight)  - qu.PauliStringSum2COO_numpy(ls,weight).diagonal()) / 2
    return -H.real


def expectation(state,hamil_diag):
    return torch.sum(state.conj() * hamil_diag * state).real

def optimality_ratio(state,min_indices):
    return torch.dot(state[min_indices].conj(), state[min_indices]).real


def expected_min_of_N_samples(P, V, N):
    """
    Calculate the expected minimum value E[min{v(X_1), ..., v(X_N)}]
    when independently sampling N times from a discrete distribution (probability P, value V).
    
    Parameters:
    -------
    P : shape=(M,)
        Probability distribution vector, requires sum(P) = 1
    V : shape=(M,)
        Value corresponding to each outcome
    N : int
        Number of samples
    
    Returns:
    -------
    E : torch.Tensor (scalar)
        Expected minimum value
    """
    # 1) Sort by value in ascending order (sort both V and P)
    sorted_V, sorted_idx = torch.sort(torch.round(V,decimals=6))
    sorted_P = P[sorted_idx]
    
    # 2) Cumulative sum from right to left: Q_array[i] = sum_{k=i}^{M-1} sorted_P[k]
    #    Q_array[i] represents the sum of probabilities for all values >= sorted_V[i]
    cumsum_rev = torch.cumsum(torch.flip(sorted_P, dims=[0]), dim=0)
    Q_array = torch.flip(cumsum_rev, dims=[0])
    # The length of Q_array is the same as sorted_V, with the i-th element being
    # the sum of all probabilities "from sorted_V[i] to the end".
    
    # 3) Find distinct values after sorting (deduplicate), denoted as distinct_vals
    #    unique_consecutive returns deduplicated values in order
    distinct_vals = torch.unique_consecutive(sorted_V)
    
    # 4) For each distinct_val, find its "first index" in sorted_V
    #    (since sorted_V is in ascending order)
    #    This way, Q[i_val] is the sum of probabilities ">= this distinct_val"
    idx = 0
    i_first_list = []
    for val in distinct_vals:
        # Move idx forward to find the position where sorted_V[idx] == val
        while sorted_V[idx] < val:
            idx += 1
        i_first_list.append(idx)
    i_first_of_val = torch.tensor(i_first_list, dtype=torch.long)
    
    # 5) Let Q_distinct[j] = Q_array[i_first_of_val[j]]
    #    which corresponds to the total probability of ">= distinct_vals[j]"
    Q_distinct = Q_array[i_first_of_val]
    
    # 6) Calculate P(Y = distinct_vals[j]):
    #    P(Y = v_j) = Q_j^N - Q_{j+1}^N  (where Q_{m+1} := 0)
    #    So first "shift Q_distinct to the right by one position", append 0 at the end
    Q_shifted = torch.cat([Q_distinct[1:], torch.zeros(1, dtype=Q_distinct.dtype)])
    p_min = Q_distinct**N - Q_shifted**N
    
    # 7) Expected value E[Y] = sum_j distinct_vals[j] * P(Y = distinct_vals[j])
    E = (distinct_vals * p_min).sum()
    
    return E.item()


def norm(state,hamil_diag=None,exp=None,dt=0.05):
    if exp is None:
        exp = expectation(state,hamil_diag)
    norm = 1 - 2 * dt * exp
    return norm.real


# solve linear equation S·a = b using gradient descent
def solve_linear_equation(S, b, a = None, lr = 0.01, n_iterations = 20000, tol = 1e-7, display = False):
    # Initialize vector a (random initialization or zero initialization)
    if a is None:
        a = torch.randn(S.shape[1], requires_grad=True)
    else:
        a = torch.tensor(a, requires_grad=True)
    
    # define optimizer
    optimizer = optim.Adam([a], lr=lr)

    # training loop
    best_loss = float('inf')
    best_a = None

    criterion = torch.nn.MSELoss()

    for i in range(n_iterations):
        Sa = torch.matmul(S, a)  # S·a
        
        # using loss: ||S·a - b||²
        # loss = torch.sum((Sa - b)**2)

        # using MSELoss
        loss = criterion(Sa, b)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # save the best result
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_a = a.detach().clone()
        
        
        # early stopping condition
        if loss.item() < tol:
            if display:
                print(f"Convergence condition reached, stopping at iteration {i+1}")
            break

    residuals = torch.matmul(S, best_a) - b
    residual_norm = torch.norm(residuals)
    if display:
        print(f"residual norm: {residual_norm.item()}")

    return best_a, residual_norm

# solve linear equation S·a = b using lstsq
def solve_linear_equation_lstsq(S, b, display = False):
    
    # Calculate least squares solution: a = pinv(S) * b
    a = torch.matmul(torch.pinverse(S), b)
    
    # Calculate residuals and print residual norm
    residuals = torch.matmul(S, a) - b
    residual_norm = torch.norm(residuals)
    if display:
        print(f"residual norm: {residual_norm.item():.6f}")
    
    return a, residual_norm


def two_qubit_rotations(num_qubits,operationList = ['Y','ZY','XY','YZ','YX']):
    idx_list = [i for i in range(num_qubits)]

    layers = []
    if num_qubits % 2 == 0:
        for _ in range(num_qubits-1):
            layers.append([[idx_list[i], idx_list[-3-i]] for i in range((num_qubits-1)//2)] + [[idx_list[-2], num_qubits-1]])
            idx_list = idx_list[-2:-1] + idx_list[:-2] + idx_list[-1:]
    else:
        for _ in range(num_qubits):
            layers.append([[idx_list[i], idx_list[-2-i]] for i in range(num_qubits//2)])
            idx_list = idx_list[-1:] + idx_list[:-1]


    operations = []

    for layer in layers:
        for operation in operationList:
            if len(operation) == 1:
                for idx in list(sum(layer,[])):
                    operations.append(operation+str(idx))
            elif len(operation) == 2:
                for i,j in layer:
                    operations.append(operation[0]+str(i)+'_'+operation[1]+str(j))
    
    return operations

def single_y_rotation(num_qubits):
    operations = ['Y'+str(i) for i in range(num_qubits)]
    return operations


def xqaoa(graph, mixer='X', reps = 1): # xqaoa ansatz
    # X -> X-mixer (ma-qaoa); Y -> Y-mixer; XY -> XY-mixer; M -> X=Y-mixer
    num_qubits = len(graph)
    idx_list = [i for i in range(num_qubits)]

    layers = []
    if num_qubits % 2 == 0:
        for _ in range(num_qubits-1):
            layers.append([[idx_list[i], idx_list[-3-i]] for i in range((num_qubits-1)//2)] + [[idx_list[-2], num_qubits-1]])
            idx_list = idx_list[-2:-1] + idx_list[:-2] + idx_list[-1:]
    else:
        for _ in range(num_qubits):
            layers.append([[idx_list[i], idx_list[-2-i]] for i in range(num_qubits//2)])
            idx_list = idx_list[-1:] + idx_list[:-1]

    operations = []
    for _ in range(reps):
        for layer in layers:
            for i,j in layer:
                if (i,j) in graph.edges():
                    operations.append('Z'+str(i)+'_'+'Z'+str(j))
        for i in range(num_qubits):
            if mixer == 'XY':
                operations.append('X'+str(i))
                operations.append('Y'+str(i))
            else:
                operations.append(mixer+str(i))

    return operations

def drop_weights(G):
    G_unweighted = G.copy()
    for _, _, data in G_unweighted.edges(data=True):
        data['weights'] = 1.0

    return G_unweighted

def ihva_from_graph(G, reps = 1): # ihva ansatz
    edge_buffer = []
    G_unweighted = drop_weights(G)
    while G_unweighted.edges:
        tree = nx.maximum_spanning_tree(G_unweighted)
        edge_buffer += list(tree.edges)
        G_unweighted.remove_edges_from(tree.edges)

    operations = []

    for r in range(reps):
        for edge in edge_buffer:
            if r % 2 == 0:
                i, j = edge
            else:
                j, i = edge
            operations.append('Z'+str(i)+'_'+'Y'+str(j))

    return operations


def get_circuit(num_qubits, init_state, operations, params):
    state = init_state
    layers = len(params)
    for layer in range(layers):
        circ = tc.Circuit(num_qubits, inputs = state)
        for idx, operation in enumerate(operations):
            pauli_list = [pauli[0] for pauli in operation.split('_')]
            qubit_list = [int(pauli[1:]) for pauli in operation.split('_')]
            if len(pauli_list) == 1:
                if pauli_list[0] == 'Y':
                    circ.ry(qubit_list[0], theta = params[layer,idx])
                elif pauli_list[0] == 'Z':
                    circ.rz(qubit_list[0], theta = params[layer,idx])
                elif pauli_list[0] == 'X':
                    circ.rx(qubit_list[0], theta = params[layer,idx])
                elif pauli_list[0] == 'M':
                    circ.rx(qubit_list[0], theta = params[layer,idx])
                    circ.ry(qubit_list[0], theta = params[layer,idx])
            else:
                for q_idx, pauli in enumerate(pauli_list):
                    if pauli == 'Y':
                        circ.rx(qubit_list[q_idx], theta = torch.pi/2)
                    elif pauli == 'X':
                        circ.h(qubit_list[q_idx])
                for n,m in [(qubit_list[i],qubit_list[i+1]) for i in range(len(qubit_list)-1)]:
                    circ.cnot(n,m)
                circ.rz(qubit_list[-1], theta = params[layer,idx])
                for n,m in [(qubit_list[i],qubit_list[i+1]) for i in range(len(qubit_list)-1)][::-1]:
                    circ.cnot(n,m)
                for q_idx, pauli in enumerate(pauli_list):
                    if pauli == 'Y':
                        circ.rx(qubit_list[q_idx], theta = -torch.pi/2)
                    elif pauli == 'X':
                        circ.h(qubit_list[q_idx])
        state = circ.state()
    return circ
