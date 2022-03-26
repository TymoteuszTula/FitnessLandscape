# hamiltonians.py

from exact_diagonalisation_code_sparse import create_states, create_hamiltonian_sparse
import numpy as np

class Hamiltonian:
    r"""Prototype of class designed for different possible Hamiltonians to be initialized."""

    def __init__(self):
        pass

    def get_init_ham(self):
        pass

class NNHamiltonian(Hamiltonian):

    def calculate_ham_sparse_in(self, L, J_onsite, J_nn, J_nnn, bc="infinite"):

        J = [[[] for i in range(L)] for j in range(L)]
        J_zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for i in range(L):
            for j in range(L):
                if i == j:
                    J[i][j] = J_onsite
                else:
                    if bc == 'finite':
                        if abs(i - j) == 1:
                            J[i][j] = J_nn
                        elif abs(i - j) == 2:
                            J[i][j] = J_nnn
                        else:
                            J[i][j] = J_zero
                    elif bc == "infinite":
                        if min(abs(i - j), abs(L + i - j), abs(L - i + j)) == 1:
                            J[i][j] = J_nn
                        elif min(abs(i - j), abs(L + i - j), abs(L - i + j)) == 2:
                            J[i][j] = J_nnn
                        else:
                            J[i][j] = J_zero

        return J

    def calculate_ham_sparse(self, L, states, h, J_onsite, J_nn, J_nnn, bc="infinite"):

        J = [[[] for i in range(L)] for j in range(L)]
        J_zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for i in range(L):
            for j in range(L):
                if i == j:
                    J[i][j] = J_onsite
                else:
                    if bc == 'finite':
                        if abs(i - j) == 1:
                            J[i][j] = J_nn
                        elif abs(i - j) == 2:
                            J[i][j] = J_nnn
                        else:
                            J[i][j] = J_zero
                    elif bc == "infinite":
                        if min(abs(i - j), abs(L + i - j), abs(L - i + j)) == 1:
                            J[i][j] = J_nn
                        elif min(abs(i - j), abs(L + i - j), abs(L - i + j)) == 2:
                            J[i][j] = J_nnn
                        else:
                            J[i][j] = J_zero

        return create_hamiltonian_sparse(states, params_input={"L": L, "J": J, "h": h})

    def __init__(self, L, h, J_onsite, J_nn, J_nnn, temp, bc="infinite"):
        self.L = L
        self.states = create_states(L)
        self.h = h
        self.temp = temp
        self.bc = bc
        self.J = self.calculate_ham_sparse_in(L, J_onsite, J_nn, J_nnn, bc)
        self.J_onsite = J_onsite
        self.J_nn = J_nn
        self.J_nnn = J_nnn

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(self.states, params_input=params)

class GeneralHamiltonian(Hamiltonian):

    def __init__(self, L, h, J, temp, bc="infinite"):
        self.L = L
        self.states = create_states(L)
        self.h = h
        self.temp = temp
        self.bc = bc
        self.J = J

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(self.states, params_input=params)

class RandomHamiltonianTI(Hamiltonian):

    def __init__(self, L, temp, bc="infinite", h=None, J=None, h_max=1, J_max=1):
        self.L = L
        self.states = create_states(L)
        if h == None:
            self.h = [h_max * np.random.rand(3)]
        else:
            self.h = h
        if J == None:
            J_init = J_max * np.random.rand(3, 3)
            J_init = 1 / 2 * J_init.conj().T @ J_init
            self.J = [[J_init]]
        else:
            self.J = J
        self.temp = temp
        self.bc = bc

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(self.states, params_input=params)

