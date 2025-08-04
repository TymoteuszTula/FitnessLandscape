# hamiltonians.py

r""" This module contains classes for different types of target 
Hamiltonians."""

from tools import create_hamiltonian_sparse
import numpy as np

class Hamiltonian:
    r""" Prototype of class designed for different possible Hamiltonians 
    to be initialized."""

    def __init__(self):
        pass

    def get_init_ham(self):
        pass

    def get_dimension(self):
        """ return the dimension of the underlying Hilbert space"""
        pass

class GeneralHamiltonian(Hamiltonian):
    r""" Creates a general Hamiltonian with long-range coupling and a field 
    strength."""

    def __init__(self, L, h, J, temp, bc="infinite"):
        self.L = L
        self.h = h
        self.temp = temp
        self.bc = bc
        self.J = J

    def get_dimension(self):
        """ return the dimension of the underlying Hilbert space"""
        return 2**self.L

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(params_input=params)

class NNHamiltonian(Hamiltonian):
    r""" Creates a nearest-neighbour Hamiltonian with long-range coupling
    and a field strength."""
    def __init__(self, L, h, J_onsite, J_nn, J_nnn, temp, bc="infinite"):
        self.L = L
        self.h = h
        self.temp = temp
        self.bc = bc
        self.J = self.calculate_ham_sparse_in(L, J_onsite, J_nn, J_nnn, bc)
        self.J_onsite = J_onsite
        self.J_nn = J_nn
        self.J_nnn = J_nnn

    def calculate_ham_sparse_in(self, L, J_onsite, J_nn, J_nnn, bc="infinite"):
        r""" Calculate and return the values of the nearest-neighbour
        Hamiltonian from given parameters - J_onsite, J_nn and J_nnn. These 
        correspond respectively to the onsite, nearest-neighbour and 
        next-nearest-neighbour interactions. This method returns just the 
        interaction matrix J."""
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
    
    def calculate_ham_sparse(self, L, h, J_onsite, J_nn, J_nnn, bc="infinite"):
        r""" Calculate and return the values of the nearest-neighbour
        Hamiltonian from given parameters - J_onsite, J_nn and J_nnn. These 
        correspond respectively to the onsite, nearest-neighbour and 
        next-nearest-neighbour interactions. This method returns full 
        Hamiltonian in sparse format."""
        J = self.calculate_ham_sparse_in(L, J_onsite, J_nn, J_nnn, bc)

        params = {"L": self.L, "J": J, "h": self.h}
        return create_hamiltonian_sparse(params_input=params)

    def get_dimension(self):
        """ return the dimension of the underlying Hilbert space"""
        return 2**self.L

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(params_input=params)    
        
class GeneralHamiltonianRandomUniform(Hamiltonian):
    r""" Creates a Hamiltonian with long-range coupling drawn from a uniform 
    distribution over -J_max .. J_max and the field strength drawn from a 
    uniform distribution -h_max_value .. h_max_value."""
    def __init__(self, L, h_max_value, J_max_value, temp, bc="infinite"):
        self.L = L
        self.temp = temp
        self.bc = bc
        # Randomly Initialise J:
        # First, we create an list of length L of empty lists:
        J = np.empty((L, 0)).tolist()

        # Then, we populate each list within the list with another list of 
        # length L:
        for i in range(L):
            J[i]=np.empty((L, 0)).tolist()

        # We now have a list of list of empty lists: J[i][j] gives the 
        # interaction between the spin at i and the spin at j 
        # (not yet populated)
        # Fill J:
        # We populate each element J[i][j] with a matrix giving the 9 
        # interactions between the x,y,z components of the spin at i and 
        # the x,y,z components of the spin at j:
        for i in range(L):
            for j in range(L):
                # uniform signed distribution of magnitudes for different 
                # spin components
                tmp_J=J_max_value*(1.0-2.0*np.random.rand(3,3)) 
                J[i][j]=0.5*(tmp_J+tmp_J.T)
		

        # Note due to the structure of the Hamiltonian and the spin commutation
        # relations, we can assume J[i][j][alpha][beta] to have certain 
        # symmetries, for instance J[i][i][alpha][beta]=-J[i][i][beta][alpha]
        # J[i][j][alpha][beta]=J[j][i][beta][alpha]. However, we do not need to
        # impose those symmetries - the Hamiltonian will take care of that. 
        # See random_hamiltonians.lyx
        #
        # Now create the random field:
        #
        # uniform signed distribution of field strength components
        self.h = [h_max_value * (1.0-2.0*np.random.rand(3))] 
        self.J = J

    def get_dimension(self):
        """ return the dimension of the underlying Hilbert space"""
        return 2**self.L

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(params_input=params)

class GeneralHamiltonianRandomUniformOverallSign(Hamiltonian):
    r""" Creates a Hamiltonian with long-range coupling drawn from a uniform 
    distribution of maximum magnitude J_max with overall sign for each link and
    the field strength drawn from a normal distribution with standard deviation
    h_sigma."""
    def __init__(self, L, h_sigma, J_max_value, temp, bc="infinite"):
        self.L = L
        self.temp = temp
        self.bc = bc
        # Randomly Initialise J:
        # First, we create an list of length L of empty lists:
        J = np.empty((L, 0)).tolist()

        # Then, we populate each list within the list with another list of 
        # length L:
        for i in range(L):
            J[i]=np.empty((L, 0)).tolist()

        # We now have a list of list of empty lists: J[i][j] gives the 
        # interaction between the spin at i and the spin at j 
        # (not yet populated)
        # Fill J:
        # We populate each element J[i][j] with a matrix giving the 9 
        # interactions between the x,y,z components of the spin at i and 
        # the x,y,z components of the spin at j:
        for i in range(L):
            for j in range(L):
                # uniform distribution of magnitudes for different spin 
                # components
                tmp_J=J_max_value*np.random.rand(3,3) 
                # symmetrize and apply overall random sign
                J[i][j]=0.5*(tmp_J+tmp_J.T)*(1.0-2.0*np.random.rand()) 

        # Note due to the structure of the Hamiltonian and the spin commutation
        # relations, we can assume J[i][j][alpha][beta] to have certain 
        # symmetries, for instance J[i][i][alpha][beta]=-J[i][i][beta][alpha]
        # J[i][j][alpha][beta]=J[j][i][beta][alpha]. However, we do not need to
        # impose those symmetries - the Hamiltonian will take care of that. 
        # See random_hamiltonians.lyx
        #
        # Now create the random field:
        #
        self.h = [h_sigma * np.random.randn(3)]
        self.J = J

    def get_dimension(self):
        """ return the dimension of the underlying Hilbert space"""
        return 2**self.L

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(params_input=params)

class GeneralHamiltonianRandomNormal(Hamiltonian):
    """ Creates a Hamiltonian with long-range coupling drawn from a gaussian 
    distribution of standard deviation J_sigma and the field strength drawn
    from a gaussian distribution of standard deviation h_sigma."""
    def __init__(self, L, h_sigma, J_sigma, temp, bc="infinite"):
        self.L = L
        self.temp = temp
        self.bc = bc
        # Randomly Initialise J:
        # First, we create an list of length L of empty lists:
        J = np.empty((L, 0)).tolist()

        # Then, we populate each list within the list with another list of 
        # length L:
        for i in range(L):
            J[i]=np.empty((L, 0)).tolist()

        # We now have a list of list of empty lists: J[i][j] gives the 
        # interaction between the spin at i and the spin at j 
        # (not yet populated) 
        # Fill J:
        # We populate each element J[i][j] with a matrix giving the 9 
        # interactions between the x,y,z components of the spin at i and 
        # the x,y,z components of the spin at j:
        for i in range(L):
            for j in range(L):
                # gaussian distribution of magnitudes for different spin 
                # components
                tmp_J=J_sigma*np.random.randn(3,3) 
                J[i][j]=0.5*(tmp_J+tmp_J.T) # symmetrize the couplings


        # Note due to the structure of the Hamiltonian and the spin commutation 
        # relations, we can assume J[i][j][alpha][beta] to have certain 
        # symmetries, for instance J[i][i][alpha][beta]=-J[i][i][beta][alpha]
        # J[i][j][alpha][beta]=J[j][i][beta][alpha]. However, we do not need to
        # impose those symmetries - the Hamiltonian will take care of that. 
        # See random_hamiltonians.lyx
        #
        # Now create the random field:
        #
        # gaussian distribution of field strength components
        self.h = [h_sigma * np.random.randn(3)] 
        self.J = J

    def get_dimension(self):
        """ return the dimension of the underlying Hilbert space"""
        return 2**self.L

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(params_input=params)

class RandomHamiltonianTI(Hamiltonian):

    def __init__(self, L, temp, bc="infinite", h=None, J=None, h_max=1, J_max=1):
        self.L = L
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

    def get_dimension(self):
        """ return the dimension of the underlying Hilbert space"""
        return 2**self.L

    def get_init_ham(self):
        params = {"L": self.L, "J": self.J, "h": self.h}
        return create_hamiltonian_sparse(params_input=params)