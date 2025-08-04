# stability_analysis_class.py

r"""This code contains class for generating and saving random hamiltonians and states, and
corresponding correlation functions to perform stability analysis of theorem 1 and 2. That is done 
for both finite and zero temperature cases.
"""

# Imports

from scipy import rand
from tools import create_hamiltonian_sparse
from tools import create_sx_sparse, create_sy_sparse, create_sz_sparse
from tools import SijCalculator
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sprs
import scipy.sparse.linalg as sprsla
import pickle
import time
import os 
import tqdm

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
    
    # example of available parameters that can be passed to functions generate_random_Sij_*
    template_params = {
        #'geometry': 'ring' # valid choices for geometry: ring or line - comment one out as desired
        'geometry': 'line'
    }

    def __init__(self, ham, randomizer, corr = ["Sxx", "Sxy", "Sxz", "Syx", "Syy", "Syz", "Szx", "Szy", "Szz"], save_rand_ham=False, temp_mul=None, temp_type="value",
                       no_qpoints=100, save_Sqs=False, save_Sijs=False, save_rham_params=False,
                       save_Sqints=False, extra_params=template_params):
        """ constructor, taking the following parameters
			ham: Hamiltonian object
			randomizer: randomizer object
			corr: list
			   set of correlators that should be calculated
			save_rand_ham: bool
			   flag indicating if randomly generated Hamiltonians are saved in output
			temp_mul: float | None
			   included in output, not currently used.
			temp_type: string
			   can be "value" - included in output, not currently used.
			no_qpoints: int
			   number of points used to discretize the BZ along each dimension
			save_Sqs: bool
			   if True, all structure factors are appended to a list at self.Sqs during run
			save_Sijs: bool
			   if True, all real-space correlators are appended to a list at self.Sijs during run
			save_Sqints: bool
			   if True, all Sq_int (sums of components of structure factor) are appended to a list at self.Sqints during run
			save_rham_params: bool
				if True, sets of parameters for generated random Hamiltonians are appended to a list at self.ham_params during run
			extra_params: dict
			   dictionary of additional parameters;
			       fields currently implemented
				      geometry: string
				        allowed values are 'line' or 'ring'
        """
        self.ham = ham
        self.randomizer = randomizer
        self.corr = corr
        self.params=extra_params

        self.dist = []
        self.diffSij = []
        self.diffSq = []
        self.diffSq_int = []
        self.en = []
        if self.randomizer.rand_ham:
            self.ham_dist = []
            if save_rand_ham:
                self.rhams = []
        self.save_rand_ham = save_rand_ham
        self.save_Sijs = save_Sijs
        self.save_Sqints = save_Sqints
        self.save_rham_params = save_rham_params
        self.time = []
        self.temp_mul = temp_mul
        self.temp_type = temp_type
        self.Sqs = []
        self.Sijs = []
        self.Sqints = []
        self.ham_params = []
        self.no_qpoints = no_qpoints
        self.save_Sqs = save_Sqs
        #self.Sq_init = []

    def set_seed(self, seed):
        np.random.seed(seed)
        time.sleep(0.1)

    def run(self, no_of_samples):
        r"""Run algorithm - generate `no_of_samples` times data and append them to correct lists.

        Parameters
        ----------
			no_of_samples: int
			   number of random samples to generate

        """

        if self.randomizer.rand_ham:
            if self.save_rand_ham:
                dist, en, diffSij, diffSq, ham_dist, hams, Sqs, Sijs, Sqints, Sq_int, NNInfos = self.generate_random_Sij_sparse_with_hams(no_of_samples)
                self.rhams += hams
                self.NNInfos = NNInfos
            else:
                dist, en, diffSij, diffSq, ham_dist, Sqs, Sijs, Sqints, Sq_int, ham_params = self.generate_random_Sij_sparse(no_of_samples)
            self.dist += dist
            self.en += en
            self.diffSij += diffSij
            self.diffSq += diffSq
            self.diffSq_int += Sq_int
            self.ham_dist += ham_dist
            if self.save_Sqs:
                self.Sqs += Sqs
            if self.save_Sijs:
                self.Sijs += Sijs
            if self.save_Sqints:
                self.Sqints += Sqints
            if self.save_rham_params:
                self.ham_params += ham_params
            
        else:
            dist, en, diffSij, diffSq, Sqs, Sijs, Sqints, Sq_int = self.generate_random_Sij_sparse(no_of_samples)
            self.dist += dist
            self.en += en
            self.diffSij += diffSij
            self.diffSq += diffSq
            self.diffSq_int += Sq_int
            if self.save_Sqs:
                self.Sqs += Sqs
            if self.save_Sijs:
                self.Sijs += Sijs
            if self.save_Sqints:
                self.Sqints += Sqints

        

    def save_random_samples(self, foldername, filename=None):
        r"""Save lists containing data: dist, diffSij, en, ham_dist to hard drive. Foldername should 
        be of form './.../' and filename should be without /.
        
        Parameters
        ----------

        """

        info = {"L": self.ham.L, "h": self.ham.h, "J": self.ham.J, "delta": self.randomizer.delta, 
                "true_temp": self.ham.temp, "bc": self.ham.bc, "rand_ham": self.randomizer.rand_ham, 
                "rand_delta": self.randomizer.rand_delta, "runtime": self.time, 
                "temp_mul": self.temp_mul, "temp_type": self.temp_type}
        if not self.randomizer.rand_ham:
            data_to_save = {"info": info, "dist": self.dist, "diffSij": self.diffSij, 
                            "diffSq": self.diffSq, "en": self.en, "Sqs": self.Sqs, "Sq_init": self.Sq_in,
                            "Sq_int": self.diffSq_int, "Sqints": self.Sqints}
        else:
            data_to_save = {"info": info, "dist": self.dist, "diffSij": self.diffSij, 
                            "diffSq": self.diffSq, "en": self.en, "ham_dist": self.ham_dist, "Sqs": self.Sqs,
                             "Sq_init": self.Sq_in, "Sq_int": self.diffSq_int, "ham_params": self.ham_params,
                             "Sqints": self.Sqints}
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


    def generate_random_Sij_sparse(self, no_of_samples, params={}):
        r"""Generate data for a given no of samples:
			This routine provides an algorithm for randomly sampling the landscape of the fitness function (deviation from target structure factor)
        Parameters
        ----------
			no_of_samples: int
			   number of random states to consider
        """
        
        #H_in = calculate_ham_sparse(L, states, h, J_onsite, J_nn, J_nnn, bc)
        #H_in = create_hamiltonian_sparse(L, params_input={"L": self.L, "J": self.J, "h": self.h})
        H_in = self.ham.get_init_ham()
        SX = create_sx_sparse(self.ham.L)
        SY = create_sy_sparse(self.ham.L)
        SZ = create_sz_sparse(self.ham.L)
        # New lines
        CorrelationCalculator = SijCalculator(self.params)
        exp_fac = CorrelationCalculator.calculate_exp_fac(self.ham.L, self.no_qpoints)
        Lambdas = CorrelationCalculator.calculate_Lambdas(self.ham.L)
        if self.ham.temp == 0:
            en_in, gs = SijCalculator.find_gs_sparse(H_in)
            state_in = gs[:,0]
            #Sij_in = SijCalculator.return_Sij(self.ham.L, state_in, SX, SY, SZ, self.ham.temp)
            Sij_in, Sq_in, Sq_int_in = SijCalculator.return_Sq2(self.ham.L, state_in, SX, SY, SZ, 
                                                       self.ham.temp, no_ofqpoints=self.no_qpoints,
                                                       exp_fac=exp_fac, Lambdas=Lambdas)
        else:
            en_in, _ = SijCalculator.find_gs_sparse(H_in)
            state_in = SijCalculator.return_dm_not_sparse(H_in, 1/self.ham.temp)
            #Sij_in = SijCalculator.return_Sij(self.ham.L, state_in, SX, SY, SZ, self.ham.temp)
            Sij_in, Sq_in, Sq_int_in = SijCalculator.returnSq2_dm_not_sparse(self.ham.L, state_in, SX, SY, SZ,
                                                       no_ofqpoints=self.no_qpoints,
                                                       exp_fac=exp_fac, Lambdas=Lambdas)

        self.Sij_in = Sij_in
        self.Sq_in = Sq_in
        self.Sq_int_in = Sq_int_in

        start_time = time.time()
        # seeds = [[np.random.randint(1000)] for i in range(self.randomizer.no_of_processes)]
        with Pool(processes=self.randomizer.no_of_processes) as pool:
            seeds = np.random.randint(1000, size=self.randomizer.no_of_processes)
            for i in range(self.randomizer.no_of_processes):
                pool.apply_async(self.set_seed, (seeds[i],))
            params = [{"H_in": H_in, "SX": SX, "SY": SY, "SZ": SZ, "en_in": en_in, 
                      "state_in": state_in, "Sij_init": Sij_in, "Sq_init": Sq_in, 
                      "corr": self.corr, "exp_fac": exp_fac, "Lambdas": Lambdas,
                      "Sq_int_in": Sq_int_in, "no_qpoints": self.no_qpoints, "return_ham": True}]
            iter = no_of_samples * params
            # seeds = np.random.randint(1000, size=4)
            # for j in range(self.randomizer.no_of_processes):
            #     iter[j]["seed"] = seeds[j]
            # print ("Input to pool: t-iter=", type(iter))
            #data = pool.map(self.randomizer.return_random_state, iter)
            data = list(tqdm.tqdm(pool.imap_unordered(self.randomizer.return_random_state, iter), total=len(iter)))
        stop_time = time.time()
        print ("Completed sampling random instances in time ", stop_time-start_time, flush=True)

        self.time.append(stop_time-start_time)

        if self.randomizer.rand_ham:
        
            #print("sample data:",data[0])
            print("sample data (keys):",data[0].keys(), flush=True)
            en = [np.real(data[i]["energy"]) for i in range(no_of_samples)]
            dist = [data[i]["dist"] for i in range(no_of_samples)]
            diffSij = [data[i]["Sij"] for i in range(no_of_samples)]
            diffSq = [data[i]["Sq"] for i in range(no_of_samples)]
            dist_ham = [data[i]["dist_ham"] for i in range(no_of_samples)]
            Sqs = [data[i]["Sq_list"] for i in range(no_of_samples)]
            Sijs = [data[i]["Sij_list"] for i in range(no_of_samples)]
            Sqints = [data[i]["Sq_int_list"] for i in range(no_of_samples)]
            Sq_int = [data[i]["Sq_int"] for i in range(no_of_samples)]
            ham_params = [data[i]["ham_params"] for i in range(no_of_samples)]
            return dist, en, diffSij, diffSq, dist_ham, Sqs, Sijs, Sqints, Sq_int, ham_params
        else:
            en = [np.real(data[i]["energy"]) for i in range(no_of_samples)]
            dist = [data[i]["dist"] for i in range(no_of_samples)]
            diffSij = [data[i]["Sij"] for i in range(no_of_samples)]
            diffSq = [data[i]["Sq"] for i in range(no_of_samples)]
            Sqs = [data[i]["Sq_list"] for i in range(no_of_samples)]
            Sijs = [data[i]["Sij_list"] for i in range(no_of_samples)]
            Sqints = [data[i]["Sq_int_list"] for i in range(no_of_samples)]
            Sq_int = [data[i]["Sq_int"] for i in range(no_of_samples)]
            return dist, en, diffSij, diffSq, Sqs, Sijs, Sqints, Sq_int

    
    def generate_random_Sij_sparse_with_hams(self, no_of_samples):
        r"""Generate data for a given no of samples
        
        Parameters
        ----------

        """
        
        #H_in = calculate_ham_sparse(L, states, h, J_onsite, J_nn, J_nnn, bc)
        #H_in = create_hamiltonian_sparse(L, params_input={"L": self.L, "J": self.J, "h": self.h})
        H_in = self.ham.get_init_ham()
        SX = create_sx_sparse(self.ham.L)
        SY = create_sy_sparse(self.ham.L)
        SZ = create_sz_sparse(self.ham.L)
        CorrelationCalculator = SijCalculator(self.params)
        exp_fac = CorrelationCalculator.calculate_exp_fac(self.ham.L, self.no_qpoints)
        Lambdas = CorrelationCalculator.calculate_Lambdas(self.ham.L)
        if self.ham.temp == 0:
            en_in, gs = SijCalculator.find_gs_sparse(H_in)
            state_in = gs[:,0]
            #Sij_in = SijCalculator.return_Sij(self.ham.L, state_in, SX, SY, SZ, self.ham.temp)
            Sij_in, Sq_in, Sq_int_in = SijCalculator.return_Sq2(self.ham.L, state_in, SX, SY, SZ, 
                                                       self.ham.temp, no_ofqpoints=self.no_qpoints,
                                                       exp_fac=exp_fac, Lambdas=Lambdas)
        else:
            en_in, _ = SijCalculator.find_gs_sparse(H_in)
            state_in = SijCalculator.return_dm_not_sparse(H_in, 1/self.ham.temp)
            #Sij_in = SijCalculator.return_Sij(self.ham.L, state_in, SX, SY, SZ, self.ham.temp)
            Sij_in, Sq_in, Sq_int_in = SijCalculator.returnSq2_dm_not_sparse(self.ham.L, state_in, SX, SY, SZ,
                                                       no_ofqpoints=self.no_qpoints, exp_fac=exp_fac,
                                                       Lambdas=Lambdas)

        self.Sij_in = Sij_in
        self.Sq_in = Sq_in
        self.Sq_int_in = Sq_int_in

        start_time = time.time()
        # seeds = [[np.random.randint(1000)] for i in range(self.randomizer.no_of_processes)]
        with Pool(processes=self.randomizer.no_of_processes) as pool:
            seeds = np.random.randint(1000, size=self.randomizer.no_of_processes)
            for i in range(self.randomizer.no_of_processes):
                pool.apply_async(self.set_seed, (seeds[i],))
            params = [{"H_in": H_in, "SX": SX, "SY": SY, "SZ": SZ, "en_in": en_in, 
                      "state_in": state_in, "Sij_init": Sij_in, "Sq_init": Sq_in, 
                      "corr": self.corr, "return_ham": True, "exp_fac": exp_fac,
                      "Lambdas": Lambdas, "Sq_int_in": Sq_int_in, "no_qpoints": self.no_qpoints}]
            iter = no_of_samples * params
            # seeds = np.random.randint(1000, size=4)
            # for j in range(self.randomizer.no_of_processes):
            #     iter[j]["seed"] = seeds[j]
            # data = pool.map(self.randomizer.return_random_state, iter)
            data = list(tqdm.tqdm(pool.imap_unordered(self.randomizer.return_random_state, iter), total=len(iter)))
        stop_time = time.time()
        print ("Completed sampling random instances in time: ",stop_time-start_time, flush=True)
        self.time.append(stop_time-start_time)
        
        en = [np.real(data[i]["energy"]) for i in range(no_of_samples)]
        dist = [data[i]["dist"] for i in range(no_of_samples)]
        diffSij = [data[i]["Sij"] for i in range(no_of_samples)]
        diffSq = [data[i]["Sq"] for i in range(no_of_samples)]
        dist_ham = [data[i]["dist_ham"] for i in range(no_of_samples)]
        rham = [data[i]["rham"] for i in range(no_of_samples)]
        NNinfos = [data[i]["NN_data"] for i in range(no_of_samples)]
        Sqs = [data[i]["Sq_list"] for i in range(no_of_samples)]
        Sq_int = [data[i]["Sq_int"] for i in range(no_of_samples)]
        Sijs = [data[i]["Sij_list"] for i in range(no_of_samples)]
        return dist, en, diffSij, diffSq, dist_ham, rham, Sqs, Sijs, Sq_int, NNinfos
    