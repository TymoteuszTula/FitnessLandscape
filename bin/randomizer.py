# randomizer.py

import numpy as np
from exact_diagonalisation_code_sparse import create_hamiltonian_sparse
from tools import SijCalculator
from hamiltonians import NNHamiltonian
import scipy.sparse as sprs
import scipy.sparse.linalg as sprsla
from math import sqrt

class Randomizer:
    r"""Prototype of class designed to chose a randomization process to generate data for stability 
    analysis."""

    def __init__(self):
        pass

    def return_random_state(self):
        """ This function should generate a random sample and then return the data for the generated instance of a random Hamiltonian.
		
		(abstract interface routine to be implemented in derived classes) """
        pass

    def calculate_dist_overlap(self, state_in, state_rand):
        dist = (np.abs(state_rand[np.newaxis].conj() @ state_in[np.newaxis].T)**2)[0,0]
        dist = 1 - dist
        # overlap = (state_rand[np.newaxis].conj() @ state_in[np.newaxis].T)[0,0]
        # dist = 2 * (1 - np.real(overlap))
        return dist

class RandomizerHamiltonianNN(Randomizer):
    """ class to create random instances of NN Hamiltonians, but modulating the total magnitude o
        the randomness with an extra uniform distribution; this yields a uniform distribution of
        given widths for different coupling parameters around the reference Hamiltonian """
    def __init__(self, ham, delta, no_of_processes):
        """ create an object to generate a sequence of NN Hamiltonian with parameters as follows
			ham: instance of the reference Hamiltonian
			
			delta: array of widths of random distributions (uniform) for:
			delta[0] = 1/2 magnitude of deviation of J_onsite from reference Hamiltonian
			delta[1] = 1/2 magnitude of deviation of J_nn from reference Hamiltonian
			delta[2] = 1/2 magnitude of deviation of J_nnn from reference Hamiltonian
			delta[2] = 1/2 magnitude of deviation of random field h from reference Hamiltonian
			
			no_of_processes: int
				number of threads to run in parallel
		"""
        self.ham = ham#
        print("Using RandomizerHamiltonianNN")
        if not isinstance(delta, list):
            raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
        elif len(delta) != 4:
            raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
        self.delta = delta
        self.no_of_processes = no_of_processes
        self.rand_ham = True
        self.rand_delta = False

    def return_random_state(self, params):
        super().return_random_state()
        delta = self.delta
        L = self.ham.L
        states = self.ham.states
        h = self.ham.h
        J_onsite = self.ham.J_onsite
        J_nn = self.ham.J_nn
        J_nnn = self.ham.J_nnn
        bc = self.ham.bc
        temp = self.ham.temp
        Sij_init = params["Sij_init"]
        Sq_init = params["Sq_init"]
        state_in = params["state_in"]
        SX = params["SX"]
        SY = params["SY"]
        SZ = params["SZ"]
        exp_fac = params["exp_fac"]
        Lambdas = params["Lambdas"]
        corr = params["corr"]
        H_in = params["H_in"]
        en_in = params["en_in"]
        Sq_int_in = params["Sq_int_in"]
        try:
            return_ham = params["return_ham"]
        except:
            return_ham = False

        ranH_J_onsite = delta[0] * (np.random.rand(3, 3) - 0.5)
        ranH_J_nn = delta[1] * (np.random.rand(3, 3) - 0.5)
        ranH_J_nnn = delta[2] * (np.random.rand(3, 3) - 0.5)
        ranH_J_onsite = 1 / 2 * ranH_J_onsite.T @ ranH_J_onsite
        ranH_J_nn = 1 / 2 * ranH_J_nn.T @ ranH_J_nn
        ranH_J_nnn = 1 / 2 * ranH_J_nnn.T @ ranH_J_nnn
        ranH_h = delta[3] * (np.random.rand(3) - 0.5)

        H_plus_delta = self.ham.calculate_ham_sparse(L, states, h + ranH_h, J_onsite + ranH_J_onsite,
                                                     J_nn + ranH_J_nn, J_nnn + ranH_J_nnn, bc=bc)
        if temp == 0:
            en_rand, gs = SijCalculator.find_gs_sparse(H_plus_delta)
            state_rand = gs[:,0]
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=100, exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)
            dist = self.calculate_dist_overlap(state_in, state_rand)
            energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] - en_in
            
        else:
            en_rand, _ = SijCalculator.find_gs_sparse(H_plus_delta)
            state_rand = SijCalculator.return_dm_sparse(H_plus_delta, 1/temp)
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=100,exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)
            dist = np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())
            energy = (H_in @ state_in).trace() - (H_in @ state_rand).trace() 

        H_in_m = H_in - en_in[0] * sprs.eye(len(states))
        H_rand_m = H_plus_delta - en_rand[0] * sprs.eye(len(states))
        dist_ham = sprsla.norm(H_in_m / H_in_m.trace() - H_rand_m / H_rand_m.trace())

        if return_ham:
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy,
                    "Sq_list": Sq_rand, "Sij_list": Sij_rand, "dist_ham": dist_ham, "rham": H_plus_delta, "Sq_int": sqrt(Sq_int_total),
                    "ham_params": {
                            "h": h + ranH_h,
                            "J_onsite": J_onsite + ranH_J_onsite,
							"J_nn": J_nn+ ranH_J_nn, "J_nnn": J_nnn + ranH_J_nnn},
							"Sq_int_list": Sq_int_rand}
        else:
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy,
                    "Sq_list": Sq_rand, "Sij_list": Sij_rand, "dist_ham": dist_ham, "Sq_int": sqrt(Sq_int_total)
                    , "Sq_int_list": Sq_int_rand}
        # previous return statement:
#        if return_ham:
#            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": 1/L/9 * sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand, "dist_ham": dist_ham, "rham": H_plus_delta, "Sq_int_list": Sq_int_rand}
#        else:
#            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": 1/L/9 * sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand, "dist_ham": dist_ham, "Sq_int_list": Sq_int_rand}
            


class RandomizerHamiltonianRandom(Randomizer):

    def __init__(self, ham, delta, no_of_processes):
        self.ham = ham
        print("Using RandomizerHamiltonianRandom")
        if not isinstance(delta, list):
            raise ValueError("If change in Hamiltonian, delta has to be a list of 2 elements")
        elif len(delta) != 2:
            raise ValueError("If change in Hamiltonian, delta has to be a list of 2 elements")
        self.delta = delta
        self.no_of_processes = no_of_processes
        self.rand_ham = True
        self.rand_delta = False

    def return_random_state(self, params):
        super().return_random_state()
        delta = self.delta
        L = self.ham.L
        states = self.ham.states
        h = self.ham.h
        J = self.ham.J
        bc = self.ham.bc
        temp = self.ham.temp
        Sij_init = params["Sij_init"]
        Sq_init = params["Sq_init"]
        state_in = params["state_in"]
        SX = params["SX"]
        SY = params["SY"]
        SZ = params["SZ"]
        corr = params["corr"]
        H_in = params["H_in"]
        en_in = params["en_in"]
        try:
            return_ham = params["return_ham"]
        except:
            return_ham = False

        ranH_J = delta[0] * (np.random.rand(3, 3) - 0.5)
        ranH_J = 1 / 2 * ranH_J.conj().T @ ranH_J
        ranH_h = delta[1] * (np.random.rand(3) - 0.5)

        H_plus_delta = create_hamiltonian_sparse(states, params_input={"L":L, "J": J + ranH_J,
                                                                       "h": h + ranH_h})
        if temp == 0:
            en_rand, gs = SijCalculator.find_gs_sparse(H_plus_delta)
            state_rand = gs[:,0]
            Sij_rand, Sq_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=100, exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            dist = self.calculate_dist_overlap(state_in, state_rand)
            energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] - en_in
            
        else:
            en_rand, _ = SijCalculator.find_gs_sparse(H_plus_delta)
            state_rand = SijCalculator.return_dm_sparse(H_plus_delta, 1/temp)
            Sij_rand, Sq_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=100, exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            dist = np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())
            energy = (H_in @ state_in).trace() - (H_in @ state_rand).trace() 

        H_in_m = H_in - en_in[0] * sprs.eye(len(states))
        H_rand_m = H_plus_delta - en_rand[0] * sprs.eye(len(states))
        dist_ham = sprsla.norm(H_in_m / H_in_m.trace() - H_rand_m / H_rand_m.trace())

        if return_ham:
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": 1/L/9 * sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand, "dist_ham": dist_ham, "rham": H_plus_delta}
        else:
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": 1/L/9 * sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand, "dist_ham": dist_ham}

class RandomizerHamiltonianRandomRandomDelta(Randomizer):

    def __init__(self, ham, delta, no_of_processes):
        self.ham = ham
        if not isinstance(delta, list):
            raise ValueError("If change in Hamiltonian, delta has to be a list of 2 elements")
        elif len(delta) != 2:
            raise ValueError("If change in Hamiltonian, delta has to be a list of 2 elements")
        self.delta = delta
        self.no_of_processes = no_of_processes
        self.rand_ham = True
        self.rand_delta = False

    def return_random_state(self, params):
        super().return_random_state()
        delta = [self.delta[0] * np.random.rand(1)[0], self.delta[1] * np.random.rand(1)[0]]
        L = self.ham.L
        states = self.ham.states
        h = self.ham.h
        J = self.ham.J
        bc = self.ham.bc
        temp = self.ham.temp
        Sij_init = params["Sij_init"]
        Sq_init = params["Sq_init"]
        state_in = params["state_in"]
        SX = params["SX"]
        SY = params["SY"]
        SZ = params["SZ"]
        corr = params["corr"]
        H_in = params["H_in"]
        en_in = params["en_in"]
        exp_fac = params["exp_fac"]
        Lambdas = params["Lambdas"]
        Sq_int_in = params["Sq_int_in"]
        no_qpoints = params["no_qpoints"]
        try:
            return_ham = params["return_ham"]
        except:
            return_ham = False

        ranH_J = delta[0] * (np.random.rand(3, 3) - 0.5)
        ranH_J = 1 / 2 * ranH_J.conj().T @ ranH_J
        ranH_h = delta[1] * (np.random.rand(3) - 0.5)

        H_plus_delta = create_hamiltonian_sparse(states, params_input={"L":L, "J": J + ranH_J,
                                                                       "h": h + ranH_h})
        if temp == 0:
            en_rand, gs = SijCalculator.find_gs_sparse(H_plus_delta)
            state_rand = gs[:,0]
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=no_qpoints,
                                                         exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)
            dist = self.calculate_dist_overlap(state_in, state_rand)
            energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] - en_in
            
        else:
            en_rand, _ = SijCalculator.find_gs_sparse(H_plus_delta)
            state_rand = SijCalculator.return_dm_not_sparse(H_plus_delta, 1/temp)
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.returnSq2_dm_not_sparse(L, state_rand, SX, SY, SZ, no_ofqpoints=no_qpoints,
                                                         exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)    
            dist = 1/2 * np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())
            energy = (H_in @ state_in).trace() - (H_in @ state_rand).trace() 

        H_in_m = H_in - en_in[0] * sprs.eye(len(states))
        H_rand_m = H_plus_delta - en_rand[0] * sprs.eye(len(states))
        dist_ham = sprsla.norm(H_in_m / H_in_m.trace() - H_rand_m / H_rand_m.trace())

        if return_ham:
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy,
                    "Sq_list": Sq_rand, "Sij_list": Sij_rand, "dist_ham": dist_ham, "rham": H_plus_delta, "Sq_int": sqrt(Sq_int_total),
                    "ham_params": {"J": J + ranH_J, "h": h + ranH_h}, "Sq_int_list": Sq_int_rand}
        else:
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy,
                    "Sq_list": Sq_rand, "Sij_list": Sij_rand, "dist_ham": dist_ham, "Sq_int": sqrt(Sq_int_total)
                    , "Sq_int_list": Sq_int_rand}

class RandomizerHamiltonianNNRandomDelta(Randomizer):
    """ class to create random instances of NN Hamiltonians, but modulating the total magnitude of
        the randomness with an extra uniform distribution; this yields a sawtooth distribution around
        the reference Hamiltonian, so will be overall more instances close to that , as compared to
        RandomizerHamiltonianNN """
    def __init__(self, ham, delta, no_of_processes):
        """ create an object to generate a sequence of NN Hamiltonian with parameters as follows
			ham: instance of the reference Hamiltonian
			
			delta: array of widths of random distributions for:
			delta[0] = 1/2 magnitude of deviation of J_onsite from reference Hamiltonian
			delta[1] = 1/2 magnitude of deviation of J_nn from reference Hamiltonian
			delta[2] = 1/2 magnitude of deviation of J_nnn from reference Hamiltonian
			delta[2] = 1/2 magnitude of deviation of random field h from reference Hamiltonian
			
			no_of_processes: int
				number of threads to run in parallel
        """
        self.ham = ham
        self.delta = delta
        if not isinstance(delta, list):
            raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
        elif len(delta) != 4:
            raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
        self.no_of_processes = no_of_processes
        self.rand_ham = True
        self.rand_delta = True

    def return_random_state(self, params):
        super().return_random_state()
        delta = [self.delta[0] * np.random.rand(1)[0], self.delta[1] * np.random.rand(1)[0],
                 self.delta[2] * np.random.rand(1)[0], self.delta[3] * np.random.rand(1)[0]]
        L = self.ham.L
        states = self.ham.states
        h = self.ham.h
        J_onsite = self.ham.J_onsite
        J_nn = self.ham.J_nn
        J_nnn = self.ham.J_nnn
        bc = self.ham.bc
        temp = self.ham.temp
        Sij_init = params["Sij_init"]
        Sq_init = params["Sq_init"]
        state_in = params["state_in"]
        SX = params["SX"]
        SY = params["SY"]
        SZ = params["SZ"]
        corr = params["corr"]
        H_in = params["H_in"]
        en_in = params["en_in"]
        exp_fac = params["exp_fac"]
        Lambdas = params["Lambdas"]
        Sq_int_in = params["Sq_int_in"]
        no_qpoints = params["no_qpoints"]
        try:
            return_ham = params["return_ham"]
        except:
            return_ham = False

        ranH_J_onsite = delta[0] * (np.random.rand(3, 3) - 0.5)
        ranH_J_nn = delta[1] * (np.random.rand(3, 3) - 0.5)
        ranH_J_nnn = delta[2] * (np.random.rand(3, 3) - 0.5)
        ranH_J_onsite = 1 / 2 * ranH_J_onsite.T @ ranH_J_onsite
        ranH_J_nn = 1 / 2 * ranH_J_nn.T @ ranH_J_nn
        ranH_J_nnn = 1 / 2 * ranH_J_nnn.T @ ranH_J_nnn
        ranH_h = delta[3] * (np.random.rand(3) - 0.5)

        H_plus_delta = self.ham.calculate_ham_sparse(L, states, h + ranH_h, J_onsite + ranH_J_onsite,
                                                     J_nn + ranH_J_nn, J_nnn + ranH_J_nnn, bc=bc)
        if temp == 0:
            en_rand, gs = SijCalculator.find_gs_sparse(H_plus_delta)
            state_rand = gs[:,0]
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=no_qpoints,
                                                          exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)
            dist = self.calculate_dist_overlap(state_in, state_rand)
            energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] - en_in
            
        else:
            en_rand, _ = SijCalculator.find_gs_sparse(H_plus_delta)
            state_rand = SijCalculator.return_dm_not_sparse(H_plus_delta, 1/temp)
            if np.sum(np.isnan(state_rand)) != 0:
                print("Not correct")
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.returnSq2_dm_not_sparse(L, state_rand, SX, SY, SZ, no_ofqpoints=no_qpoints,
                                                          exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)
            dist = 1/2 * np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())
            energy = (H_in @ state_in).trace() - (H_in @ state_rand).trace() 

        H_in_m = H_in - en_in[0] * sprs.eye(len(states))
        H_rand_m = H_plus_delta - en_rand[0] * sprs.eye(len(states))
        dist_ham = sprsla.norm(H_in_m / H_in_m.trace() - H_rand_m / H_rand_m.trace())

        if return_ham:
            NN_data = {"h": h + ranH_h, "J_onsite": J_onsite + ranH_J_onsite,
                       "J_nn": J_nn + ranH_J_nn, "J_nnn": J_nnn + ranH_J_nnn}
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy,
                    "Sq_list": Sq_rand, "Sij_list": Sij_rand, "dist_ham": dist_ham, "rham": H_plus_delta, "Sq_int": sqrt(Sq_int_total),
                    "ham_params": NN_data, "Sq_int_list": Sq_int_rand}
        else:
            NN_data = {"h": h + ranH_h, "J_onsite": J_onsite + ranH_J_onsite,
                       "J_nn": J_nn + ranH_J_nn, "J_nnn": J_nnn + ranH_J_nnn}
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy,
                    "Sq_list": Sq_rand, "Sij_list": Sij_rand, "dist_ham": dist_ham, "Sq_int": sqrt(Sq_int_total),
                    "ham_params": NN_data, "Sq_int_list": Sq_int_rand}


class RandomizerState(Randomizer):

    def __init__(self, ham, delta, no_of_processes):
        self.ham = ham
        self.delta = delta
        self.no_of_processes = no_of_processes
        self.rand_ham = False
        self.rand_delta = False


    def return_random_state(self, params):
        super().return_random_state()
        delta = self.delta
        L = self.ham.L
        temp = self.ham.temp
        Sij_init = params["Sij_init"]
        Sq_init = params["Sq_init"]
        state_in = params["state_in"]
        SX = params["SX"]
        SY = params["SY"]
        SZ = params["SZ"]
        corr = params["corr"]
        H_in = params["H_in"]
        en_in = params["en_in"]

        state_no = 2 ** L

        if temp == 0:
            state_rand = state_in + delta * (np.random.rand(state_no) + 1j * np.random.rand(state_no) - 0.5 * (1 + 1j))
            state_rand = state_rand / np.sqrt(np.sum(state_rand.conj() * state_rand))
            Sij_rand, Sq_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=100, exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            #dist = np.sum(np.abs(state_rand.conj() * state_in)**2)
            dist = self.calculate_dist_overlap(state_in, state_rand)
            energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] - en_in
        else:
            # Here state_in must be dense
            ranp = delta * (np.random.rand(state_no, state_no) - 0.5 + 1j * np.random.rand(state_no, state_no) -0.5j)
            ranp = ranp.conj().T @ ranp
            state_rand = state_in + ranp
            state_rand = state_rand / np.trace(state_rand)
            Sij_rand, Sq_rand = SijCalculator.return_Sq2_dm_not_sparse(L, state_rand, SX, SY, SZ, no_ofqpoints=100, )
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            dist = np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())[0,0]
            energy = (H_in @ state_rand).trace()

        return {"Sij": 1/L/9 * sqrt(S_total), "Sq": 1/L/9 * sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand}

class RandomizerStateRandomDelta(Randomizer):

    def __init__(self, ham, delta_max, no_of_processes):
        self.ham = ham
        self.delta = delta_max
        self.no_of_processes = no_of_processes
        self.rand_ham = False
        self.rand_delta = True

    def return_random_state(self, params):
        super().return_random_state()
        delta = self.delta * np.random.rand(1)
        L = self.ham.L
        temp = self.ham.temp
        Sij_init = params["Sij_init"]
        Sq_init = params["Sq_init"]
        state_in = params["state_in"]
        SX = params["SX"]
        SY = params["SY"]
        SZ = params["SZ"]
        corr = params["corr"]
        H_in = params["H_in"]
        en_in = params["en_in"]
        exp_fac = params["exp_fac"]
        Lambdas = params["Lambdas"]
        Sq_int_in = params["Sq_int_in"]
        no_qpoints = params["no_qpoints"]

        state_no = 2 ** L

        if temp == 0:
            state_rand = state_in + delta * (np.random.rand(state_no) + 1j * np.random.rand(state_no) - 0.5 * (1 + 1j))
            state_rand = state_rand / np.sqrt(np.sum(state_rand.conj() * state_rand))
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=no_qpoints,
                                                          exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)
            #dist = np.sum(np.abs(state_rand.conj() * state_in)**2)
            dist = self.calculate_dist_overlap(state_in, state_rand)
            energy = (state_rand[np.newaxis].conj() @ H_in @ state_rand[np.newaxis].T)[0,0] - en_in
        else:
            # Here state_in must be dense
            ranp = delta * (np.random.rand(state_no, state_no) - 0.5 + 1j * np.random.rand(state_no, state_no) -0.5j)
            ranp = ranp.conj().T @ ranp
            state_rand = state_in + ranp
            state_rand = state_rand / np.trace(state_rand)
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.returnSq2_dm_not_sparse(L, state_rand, SX, SY, SZ, no_ofqpoints=no_qpoints,
                                                                        exp_fac=exp_fac, Lambdas=Lambdas)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)
            dist = 1/2 * np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())[0,0]
            energy = (H_in @ state_rand).trace()

        return {"Sij": 1/L/9 * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand,
                    "Sij_list": Sij_rand, "Sq_int": Sq_int_total, "Sq_int_list": Sq_int_rand}
 
