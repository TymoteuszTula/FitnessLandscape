# stability_analysis_class.py

r"""This code contains class for generating and saving random hamiltonians and states, and
corresponding correlation functions to perform stability analysis of theorem 1 and 2. That is done 
for both finite and zero temperature cases.
"""

# Imports

from scipy import rand
from exact_diagonalisation_code_sparse import create_states, create_hamiltonian_sparse
from exact_diagonalisation_code_sparse import create_sx_sparse, create_sy_sparse, create_sz_sparse
from tools import SijCalculator
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sprs
import scipy.sparse.linalg as sprsla
import pickle
import time
import os 


class StabilityAnalysisSparse:
    r"""Class which provides or saves data for stability analysis of connection between correlation
    functions and Hamiltonians/states. By providing initial Hamiltonian, it generates random 
    Hamiltonians or states with, centred around those initial ones, and evaluates correlation 
    functions, metrics and energies/free energies.

    Parameters
    ----------

    Attributes
    ----------
    
    """

    def __init__(self, ham, randomizer, corr, save_rand_ham=False):
        self.ham = ham
        self.randomizer = randomizer
        self.corr = corr

        self.dist = []
        self.diffSij = []
        self.diffSq = []
        self.en = []
        if self.randomizer.rand_ham:
            self.ham_dist = []
            if save_rand_ham:
                self.rhams = []
        self.save_rand_ham = save_rand_ham
        self.time = []

    def run(self, no_of_samples):
        r"""Run algorithm - generate `no_of_samples` times data and append them to correct lists.

        Parameters
        ----------

        """

        if self.randomizer.rand_ham:
            if self.save_rand_ham:
                dist, en, diffSij, diffSq, ham_dist, hams = self.generate_random_Sij_sparse_with_hams(no_of_samples)
                self.rhams += hams
            else:
                dist, en, diffSij, diffSq, ham_dist = self.generate_random_Sij_sparse(no_of_samples)
            self.dist += dist
            self.en += en
            self.diffSij += diffSij
            self.diffSq += diffSq
            self.ham_dist += ham_dist
            
        else:
            dist, en, diffSij, diffSq = self.generate_random_Sij_sparse(no_of_samples)
            self.dist += dist
            self.en += en
            self.diffSij += diffSij
            self.diffSq += diffSq

        

    def save_random_samples(self, foldername, filename=None):
        r"""Save lists containing data: dist, diffSij, en, ham_dist to hard drive. Foldername should 
        be of form './.../' and filename should be without /.
        
        Parameters
        ----------

        """

        info = {"L": self.ham.L, "h": self.ham.h, "J": self.ham.J, "delta": self.randomizer.delta, 
                "temp": self.ham.temp, "bc": self.ham.bc, "rand_ham": self.randomizer.rand_ham, 
                "rand_delta": self.randomizer.rand_delta, "runtime": self.time}
        if not self.randomizer.rand_ham:
            data_to_save = {"info": info, "dist": self.dist, "diffSij": self.diffSij, 
                            "diffSq": self.diffSq, "en": self.en}
        else:
            data_to_save = {"info": info, "dist": self.dist, "diffSij": self.diffSij, 
                            "diffSq": self.diffSq, "en": self.en, "ham_dist": self.ham_dist}
            if self.save_rand_ham:
                data_to_save["rhams"] = self.rhams

        # Create right folders
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        if filename == None:
            filename = "data.pickle"

        with open(foldername + filename, 'wb') as fh:
            pickle.dump(data_to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load_random_samples(self, foldername, filename=None):
        r"""Load lists containing data: dist, diffSij, en, ham_dist to hard drive. Foldername should 
        be of form './.../' and filename should be without /.
        
        Parameters
        ----------

        """

        if filename == None:
            filename = "data.pickle"

        with open(foldername + filename, 'rb') as fh:
            data_to_load = pickle.load(fh)

        return data_to_load


    def generate_random_Sij_sparse(self, no_of_samples):
        r"""Generate data for a given no of samples
        
        Parameters
        ----------

        """
        
        states = self.ham.states
        #H_in = calculate_ham_sparse(L, states, h, J_onsite, J_nn, J_nnn, bc)
        #H_in = create_hamiltonian_sparse(L, params_input={"L": self.L, "J": self.J, "h": self.h})
        H_in = self.ham.get_init_ham()
        SX = create_sx_sparse(states, self.ham.L)
        SY = create_sy_sparse(states, self.ham.L)
        SZ = create_sz_sparse(states, self.ham.L)
        if self.ham.temp == 0:
            en_in, gs = SijCalculator.find_gs_sparse(H_in)
            state_in = gs[:,0]
            #Sij_in = SijCalculator.return_Sij(self.ham.L, state_in, SX, SY, SZ, self.ham.temp)
            Sij_in, Sq_in = SijCalculator.return_SijSq(self.ham.L, state_in, SX, SY, SZ, 
                                                       self.ham.temp)
        else:
            en_in, _ = SijCalculator.find_gs_sparse(H_in)
            state_in = SijCalculator.return_dm_sparse(H_in, 1/self.ham.temp)
            #Sij_in = SijCalculator.return_Sij(self.ham.L, state_in, SX, SY, SZ, self.ham.temp)
            Sij_in, Sq_in = SijCalculator.return_SijSq(self.ham.L, state_in, SX, SY, SZ,
                                                       self.ham.temp)

        self.Sij_in = Sij_in
        self.Sq_in = Sq_in

        start_time = time.time()
        with Pool(processes=self.randomizer.no_of_processes) as pool:
            params = [{"H_in": H_in, "SX": SX, "SY": SY, "SZ": SZ, "en_in": en_in, 
                      "state_in": state_in, "Sij_init": Sij_in, "Sq_init": Sq_in, 
                      "corr": self.corr}]
            iter = no_of_samples * params
            data = pool.map(self.randomizer.return_random_state, iter)
        stop_time = time.time()

        self.time.append(stop_time-start_time)

        if self.randomizer.rand_ham:
            en = [data[i]["energy"] for i in range(no_of_samples)]
            dist = [data[i]["dist"] for i in range(no_of_samples)]
            diffSij = [data[i]["Sij"] for i in range(no_of_samples)]
            diffSq = [data[i]["Sq"] for i in range(no_of_samples)]
            dist_ham = [data[i]["dist_ham"] for i in range(no_of_samples)]
            return dist, en, diffSij, diffSq, dist_ham
        else:
            en = [data[i]["energy"] for i in range(no_of_samples)]
            dist = [data[i]["dist"] for i in range(no_of_samples)]
            diffSij = [data[i]["Sij"] for i in range(no_of_samples)]
            diffSq = [data[i]["Sq"] for i in range(no_of_samples)]
            return dist, en, diffSij, diffSq

    
    def generate_random_Sij_sparse_with_hams(self, no_of_samples):
        r"""Generate data for a given no of samples
        
        Parameters
        ----------

        """
        
        states = self.ham.states
        #H_in = calculate_ham_sparse(L, states, h, J_onsite, J_nn, J_nnn, bc)
        #H_in = create_hamiltonian_sparse(L, params_input={"L": self.L, "J": self.J, "h": self.h})
        H_in = self.ham.get_init_ham()
        SX = create_sx_sparse(states, self.ham.L)
        SY = create_sy_sparse(states, self.ham.L)
        SZ = create_sz_sparse(states, self.ham.L)
        if self.ham.temp == 0:
            en_in, gs = SijCalculator.find_gs_sparse(H_in)
            state_in = gs[:,0]
            #Sij_in = SijCalculator.return_Sij(self.ham.L, state_in, SX, SY, SZ, self.ham.temp)
            Sij_in, Sq_in = SijCalculator.return_SijSq(self.ham.L, state_in, SX, SY, SZ, 
                                                       self.ham.temp)
        else:
            en_in, _ = SijCalculator.find_gs_sparse(H_in)
            state_in = SijCalculator.return_dm_sparse(H_in, 1/self.ham.temp)
            #Sij_in = SijCalculator.return_Sij(self.ham.L, state_in, SX, SY, SZ, self.ham.temp)
            Sij_in, Sq_in = SijCalculator.return_SijSq(self.ham.L, state_in, SX, SY, SZ,
                                                       self.ham.temp)

        self.Sij_in = Sij_in
        self.Sq_in = Sq_in

        start_time = time.time()
        with Pool(processes=self.randomizer.no_of_processes) as pool:
            params = [{"H_in": H_in, "SX": SX, "SY": SY, "SZ": SZ, "en_in": en_in, 
                      "state_in": state_in, "Sij_init": Sij_in, "Sq_init": Sq_in, 
                      "corr": self.corr, "return_ham": True}]
            iter = no_of_samples * params
            data = pool.map(self.randomizer.return_random_state, iter)
        stop_time = time.time()

        self.time.append(stop_time-start_time)

        en = [data[i]["energy"] for i in range(no_of_samples)]
        dist = [data[i]["dist"] for i in range(no_of_samples)]
        diffSij = [data[i]["Sij"] for i in range(no_of_samples)]
        diffSq = [data[i]["Sq"] for i in range(no_of_samples)]
        dist_ham = [data[i]["dist_ham"] for i in range(no_of_samples)]
        rham = [data[i]["rham"] for i in range(no_of_samples)]
        return dist, en, diffSij, diffSq, dist_ham, rham
    

        # if self.randomizer.rand_ham:
        #     if not isinstance(self.delta, list):
        #         raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
        #     elif len(self.delta) != 4:
        #         raise ValueError("If change in Hamiltonian, delta has to be a list of 4 elements")
            
        #     with Pool(processes=self.no_of_processes) as pool:
        #         params = {"delta": delta, "L": L, "states": states, "h": h, "J_onsite": J_onsite,
        #                 "J_nn": J_nn, "J_nnn": J_nnn, "bc": bc, "temp": temp, 
        #                 "Sij_init": Sij_in, "state_in": state_in, "SX": SX, "SY": SY, "SZ": SZ,
        #                 "corr": corr, "H_in": H_in, "en_in": en_in}
        #         iter = no_of_samples * [params]
        #         data = pool.map(generate_random_parallel_ham, iter)
        #         en = [data[i]["energy"] for i in range(no_of_samples)]
        #         dist = [data[i]["dist"] for i in range(no_of_samples)]
        #         diffSij = [data[i]["Sij"] for i in range(no_of_samples)]
        #         dist_ham = [data[i]["dist_ham"] for i in range(no_of_samples)]

        #         return en, dist, diffSij, dist_ham

        # else:
        #     with Pool(processes=no_of_processes) as pool:
        #         # Change dm to dense in this case
        #         if temp != 0:
        #             state_in = state_in.todense()
        #         if rand_delta:
        #             #delta_r = delta * np.random.rand(no_of_samples)
        #             #iter = [(delta_r[i], L, temp, Sij_in, state_in, SX, SY, SZ, corr, H_in) for i in range(no_of_samples)]
        #             # iter = (delta * np.random.rand(no_of_samples), no_of_samples * [L], no_of_samples * [temp], 
        #             #         no_of_samples * [Sij_in], no_of_samples * state_in, no_of_samples * SX,
        #             #         no_of_samples * [SY], no_of_samples * [SZ], no_of_samples * [corr],
        #             #         no_of_samples * [corr], no_of_samples * [H_in])
        #             iter = [{"delta": delta * random(), "L": L, "temp": temp, 
        #                         "Sij_init": Sij_in, "state_in": state_in, "SX": SX, "SY": SY, "SZ": SZ,
        #                         "corr": corr, "H_in": H_in, "en_in": en_in} for i in range(no_of_samples)]
        #         else:
        #             #iter = no_of_samples * [(delta, L, temp, Sij_in, state_in, SX, SY, SZ, corr, H_in)]
        #             # iter = (no_of_samples * [delta], no_of_samples * [L], no_of_samples * [temp], 
        #             #         no_of_samples * [Sij_in], no_of_samples * state_in, no_of_samples * SX,
        #             #         no_of_samples * [SY], no_of_samples * [SZ], no_of_samples * [corr],
        #             #         no_of_samples * [corr], no_of_samples * [H_in])
        #             params = {"delta": delta , "L": L, "temp": temp, 
        #                         "Sij_init": Sij_in, "state_in": state_in, "SX": SX, "SY": SY, "SZ": SZ,
        #                         "corr": corr, "H_in": H_in, "en_in": en_in}
        #             iter = no_of_samples * [params]
        #         data = pool.map(generate_random_parallel_state, iter)
        #         en = [data[i]["energy"] for i in range(no_of_samples)]
        #         dist = [data[i]["dist"] for i in range(no_of_samples)]
        #         diffSij = [data[i]["Sij"] for i in range(no_of_samples)]

        #         return en, dist, diffSij
