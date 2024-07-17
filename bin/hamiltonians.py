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
        
        
class GeneralHamiltonianRandomUniform(Hamiltonian):
    """ Creates a Hamiltonian with long-range coupling drawn from a uniform distribution over -J_max .. J_max and the field strength drawn from a uniform distribution """
    def __init__(self, L, h_max_value, J_max_value, temp, bc="infinite"):
        self.L = L
        self.states = create_states(L)
        self.temp = temp
        self.bc = bc
        # Randomly Initialise J:
        # First, we create an list of length L of empty lists:
        J = np.empty((L, 0)).tolist()


        # Then, we populate each list within the list with another list of length L:
        for i in range(L):
            J[i]=np.empty((L, 0)).tolist()

        # We now have a list of list of empty lists: J[i][j] gives the interaction
        #                              between the spin at i and the spin at j (not yet populated)
        # Fill J:
        # We populate each element J[i][j] with a matrix giving the 9 interactions between
        # the x,y,z components of the spin at i and the x,y,z components of the spin at j:
        for i in range(L):
            for j in range(L):
                tmp_J=J_max_value*(1.0-2.0*np.random.rand(3,3)) # <-- uniform signed distribution of magnitudes for different spin components
                J[i][j]=0.5*(tmp_J+tmp_J.T)
		

        # Note due to the structure of the Hamiltonian and the spin commutation relations,
        # we can assume J[i][j][alpha][beta] to have certain symmetries, for instance
        # J[i][i][alpha][beta]=-J[i][i][beta][alpha]
        # J[i][j][alpha][beta]=J[j][i][beta][alpha]
        # However, we do not need to impose those symmetries - the Hamiltonian will take
        # care of that. See random_hamiltonians.lyx
        #
        # Now create the random field:
        #
        self.h = [h_max_value * (1.0-2.0*np.random.rand(3))] # <-- uniform signed distribution of field strength components
        self.J = J

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(self.states, params_input=params)


class GeneralHamiltonianRandomUniformOverallSign(Hamiltonian):
    """ Creates a Hamiltonian with long-range coupling drawn from a uniform distribution of maximum magnitude J_max with overall sign for each link and the field strength drawn from a normal distribution """
    def __init__(self, L, h_max_value, J_max_value, temp, bc="infinite"):
        self.L = L
        self.states = create_states(L)
        self.temp = temp
        self.bc = bc
        # Randomly Initialise J:
        # First, we create an list of length L of empty lists:
        J = np.empty((L, 0)).tolist()


        # Then, we populate each list within the list with another list of length L:
        for i in range(L):
            J[i]=np.empty((L, 0)).tolist()

        # We now have a list of list of empty lists: J[i][j] gives the interaction
        #                              between the spin at i and the spin at j (not yet populated)
        # Fill J:
        # We populate each element J[i][j] with a matrix giving the 9 interactions between
        # the x,y,z components of the spin at i and the x,y,z components of the spin at j:
        for i in range(L):
            for j in range(L):
                tmp_J=J_max_value*np.random.rand(3,3) # <-- uniform distribution of magnitudes for different spin components;
                J[i][j]=0.5*(tmp_J+tmp_J.T)*(1.0-2.0*np.random.rand()) # symmetrize and apply overall random sign


        # Note due to the structure of the Hamiltonian and the spin commutation relations,
        # we can assume J[i][j][alpha][beta] to have certain symmetries, for instance
        # J[i][i][alpha][beta]=-J[i][i][beta][alpha]
        # J[i][j][alpha][beta]=J[j][i][beta][alpha]
        # However, we do not need to impose those symmetries - the Hamiltonian will take
        # care of that. See random_hamiltonians.lyx
        #
        # Now create the random field:
        #
        self.h = [h_max_value * np.random.randn(3)]
        self.J = J

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(self.states, params_input=params)

class GeneralHamiltonianRandomNormal(Hamiltonian):
    """ Creates a Hamiltonian with long-range coupling drawn from a uniform distribution over -J_max .. J_max and the field strength drawn from a uniform distribution """
    def __init__(self, L, h_max_value, J_max_value, temp, bc="infinite"):
        self.L = L
        self.states = create_states(L)
        self.temp = temp
        self.bc = bc
        # Randomly Initialise J:
        # First, we create an list of length L of empty lists:
        J = np.empty((L, 0)).tolist()


        # Then, we populate each list within the list with another list of length L:
        for i in range(L):
            J[i]=np.empty((L, 0)).tolist()

        # We now have a list of list of empty lists: J[i][j] gives the interaction
        #                              between the spin at i and the spin at j (not yet populated)
        # Fill J:
        # We populate each element J[i][j] with a matrix giving the 9 interactions between
        # the x,y,z components of the spin at i and the x,y,z components of the spin at j:
        for i in range(L):
            for j in range(L):
                tmp_J=J_max_value*np.random.randn(3,3) # <-- uniform signed distribution of magnitudes for different spin components
                J[i][j]=0.5*(tmp_J+tmp_J.T) # symmetrize the couplings


        # Note due to the structure of the Hamiltonian and the spin commutation relations,
        # we can assume J[i][j][alpha][beta] to have certain symmetries, for instance
        # J[i][i][alpha][beta]=-J[i][i][beta][alpha]
        # J[i][j][alpha][beta]=J[j][i][beta][alpha]
        # However, we do not need to impose those symmetries - the Hamiltonian will take
        # care of that. See random_hamiltonians.lyx
        #
        # Now create the random field:
        #
        self.h = [h_max_value * np.random.randn(3)] # <-- uniform signed distribution of field strength components
        self.J = J

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(self.states, params_input=params)

class RandomHamiltonianTI(Hamiltonian):

    def __init__(self, L, temp, bc="infinite", h=None, J=None, h_max=1, J_max=1):
        self.L = L
        self.states = create_states(L)
        if h == None:
            self.h = [h_max * np.random.randn(3)]
        else:
            self.h = h
        if J == None:
            J_init = J_max * np.random.randn(3, 3)
            J_init = 1 / 2 * J_init.conj().T @ J_init
            self.J = [[J_init]]
        else:
            self.J = J
        self.temp = temp
        self.bc = bc

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(self.states, params_input=params)

