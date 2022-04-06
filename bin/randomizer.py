# randomizer.py

import numpy as np
from exact_diagonalisation_code_sparse import create_hamiltonian_sparse
from tools import SijCalculator
from hamiltonians import NNHamiltonian
import scipy.sparse as sprs
import scipy.sparse.linalg as sprsla

class Randomizer:
    r"""Prototype of class designed to chose a randomization process to generate data for stability 
    analysis."""

    def __init__(self):
        pass

    def return_random_state(self):
        pass

    def calculate_dist_overlap(self, state_in, state_rand):
        dist = (np.abs(state_rand[np.newaxis].conj() @ state_in[np.newaxis].T)**2)[0,0]
        dist = 1 - dist
        # overlap = (state_rand[np.newaxis].conj() @ state_in[np.newaxis].T)[0,0]
        # dist = 2 * (1 - np.real(overlap))
        return dist

class RandomizerHamiltonianNN(Randomizer):

    def __init__(self, ham, delta, no_of_processes):
        self.ham = ham
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
        corr = params["corr"]
        H_in = params["H_in"]
        en_in = params["en_in"]
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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy, "dist_ham": dist_ham,
                    "rham": H_plus_delta}
        else:
            return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy, "dist_ham": dist_ham}

class RandomizerHamiltonianRandom(Randomizer):

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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy, "dist_ham": dist_ham,
                    "rham": H_plus_delta}
        else:
            return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy, "dist_ham": dist_ham}

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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy, "dist_ham": dist_ham,
                    "rham": H_plus_delta}
        else:
            return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy, "dist_ham": dist_ham}

class RandomizerHamiltonianNNRandomDelta(Randomizer):

    def __init__(self, ham, delta, no_of_processes):
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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy, "dist_ham": dist_ham,
                        "rham": H_plus_delta}
        else:
            return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy, "dist_ham": dist_ham}


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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq_dm_not_sparse(L, state_rand, SX, SY, SZ)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            dist = np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())[0,0]
            energy = (H_in @ state_rand).trace()

        return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy}

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

        state_no = 2 ** L

        if temp == 0:
            state_rand = state_in + delta * (np.random.rand(state_no) + 1j * np.random.rand(state_no) - 0.5 * (1 + 1j))
            state_rand = state_rand / np.sqrt(np.sum(state_rand.conj() * state_rand))
            Sij_rand, Sq_rand = SijCalculator.return_SijSq(L, state_rand, SX, SY, SZ, temp)
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
            Sij_rand, Sq_rand = SijCalculator.return_SijSq_dm_not_sparse(L, state_rand, SX, SY, SZ)
            S_total = 0
            Sq_total = 0
            for corr_i in corr:
                S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
                Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
            dist = np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())[0,0]
            energy = (H_in @ state_rand).trace()

        return {"Sij": S_total, "Sq": Sq_total, "dist": dist, "energy": energy}
