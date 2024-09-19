# randomizer.py

import numpy as np
from exact_diagonalisation_code_sparse import create_hamiltonian_sparse
from tools import SijCalculator
from hamiltonians import NNHamiltonian
import scipy.sparse as sprs
import scipy.sparse.linalg as sprsla
from math import sqrt


class AbstractElementDistribution:
	"""Abstract base class defining interface for distributions generating elements for the density matrix / ground state"""
	def __init__(self):
		pass
	def get_random_elements(self, delta, shape):
		""" this method implements the call to the relevant distribution, returning a vector of corresponding random numbers of given shape
		shape: tuple
			dimensions of the required array of random numbers"""
		pass
		
class ComplexUniformDistributionSquare(AbstractElementDistribution):
	"""implementing a complex distribution which generates complex numbers in the square spanning -delta/2 .. delta/2 along the real and imaginary coordinate axis"""
	def __init__(self):
		print ("Using ComplexUniformDistributionSquare", flush=True)
		pass
	def get_random_elements(self, delta, shape):
		""" this method implements the call to the relevant distribution"""
		return delta * (np.random.rand(*shape) - 0.5 + 1j * np.random.rand(*shape) -0.5j)

class ComplexUniformDistributionCircle(AbstractElementDistribution):
	"""implementing a complex distribution which generates complex numbers in the circle of radius delta """
	def __init__(self):
		print ("Using ComplexUniformDistributionCircle", flush=True)
		pass
	def get_random_elements(self, delta, shape):
		""" this method implements the call to the relevant distribution"""
	
		angles = np.random.uniform(0, 2 * np.pi, shape)
		# Generate random radii with the correct distribution
		radii = np.sqrt(np.random.uniform(0, delta**2, shape))
		return radii * np.exp(1j * angles)
		#rst = radii * np.exp(1j * angles)
		#print("Random draw in UC: ",type(rst),rst)
		#return rst
		
		
class ComplexNormalDistribution(AbstractElementDistribution):
	"""implementing a normal distribution of standard deviation delta """
	def __init__(self):
		pass
	def get_random_elements(self, delta, shape):
		""" this method implements the call to the relevant distribution"""
		return np.random.normal(loc=0.0, scale=delta, size=shape) + 1j * np.random.normal(loc=0.0, scale=delta, size=shape)
		
		
def SelectElementDistribution(shape):
	""" select an element distribution from one of the know types:
		shape : string
			allowed values: 'uniformSquare', 'uniformCircle', or 'normal'
	"""
	if (shape== 'uniformSquare'):
		elementdistribution = ComplexUniformDistributionSquare()
	elif(shape=='uniformCircle'):
		elementdistribution = ComplexUniformDistributionCircle()
	elif(shape== 'normal'):
		elementdistribution = ComplexNormalDistribution()
	else:
		print("Unknown 'shape' of distribution, please select from: 'uniformSquare', 'uniformCircle', or 'normal' ")
		exit(1)

	return elementdistribution
	
	
class AbstractDensityMatrixDistribution:
	"""Abstract base class defining interface for distributions of density matrices"""
	def __init__(self):
		pass
	def get_density_matrix(self, reference_rho, delta):
		""" this method generates a variation around a reference density matric reference_rho, with deviation parameter delta
		
			  reference_rho: np.array
			  delta: magnitude of deviation (implementation left to derived classes)
		"""
		pass
		
		
class RSquaredDensityMatrixDistribution(AbstractDensityMatrixDistribution):
	"""Distributions of random density matrices using deviations in form of squares of random matrices"""
	def __init__(self, elementdistribution):
		self.elementdistribution = elementdistribution
	def get_density_matrix(self, reference_rho, delta):
		""" this method generates a variation around a reference density matric reference_rho, with deviation parameter delta
			reference_rho: np.array
			delta: magnitude of deviation - deviation parameter passed to element distribution
		"""
		ranp = self.elementdistribution.get_random_elements(delta, reference_rho.shape)
		ranp = ranp.conj().T @ ranp
		rho_rand = reference_rho + ranp
		return rho_rand / np.trace(rho_rand)
		
class ProjectedDensityMatrixDistribution(AbstractDensityMatrixDistribution):
	"""Distributions of random density matrices generating deviations satisfying the constraints from eliminating negative eigenvalues in the eigenbasis"""
	def __init__(self, elementdistribution, params=None):
		self.elementdistribution = elementdistribution
		if (params is not None ) and ('min_ev' in params):
			self.min_ev=np.max(min_ev,0.0)
		else:
			self.min_ev=0.0
		
	def get_density_matrix(self, reference_rho, delta):
		""" this method generates a variation around a reference density matric reference_rho, with deviation parameter delta
			reference_rho: np.array
			delta: magnitude of deviation - deviation parameter passed to element distribution
		"""
		ranp = self.elementdistribution.get_random_elements(delta, reference_rho.shape)
		ranp = 0.5*(ranp.conj().T + ranp)
		    
		rho_rand = reference_rho + ranp
		evals, evecs = np.linalg.eigh(rho_rand)
		# Ensure eigenvalues are positive
		evals = np.maximum(evals, self.min_ev)
    
		# Normalize to ensure trace is one
		evals /= np.sum(evals)
		# Reconstruct the matrix
		rho_rand = (evecs @ np.diag(evals) ) @ evecs.conj().T

		return rho_rand
		
		
def SelectDensityMatrixSampler(elementdistribution, params=None):
	""" select a method to generate a positive definite and trace 1 matrix that forms a valid density matrix:
		params : dict - defining parameters of the distribution, important keywords are
		    'method' : string
			   allowed values: 'squareRandom', 'projectEigenvalues'
		elementdistribution : AbstractElementDistribution
		    pointer to the distribution to be used for individual elements
	"""
	if not 'method' in params:
		print("Please define the 'method' of density matrix sampler for finite T calculations, choosing from: squareRandom', 'projectEigenvalues' ")
	method = params['method']
	if (method== 'squareRandom'):
		distrib = RSquaredDensityMatrixDistribution(elementdistribution)
	elif(method=='projectEigenvalues'):
		distrib = ProjectedDensityMatrixDistribution(elementdistribution, params)
	else:
		print("Unknown 'method' for generating density matrices. Please select from: 'squareRandom', 'projectEigenvalues' ")

	return distrib
		
class Randomizer:
    r"""Prototype of class designed to chose a randomization process to generate data for stability 
    analysis."""

    def __init__(self):
        pass

    def return_random_state(self):
        """ This function should generate a random sample and then return the data for the generated instance of a random Hamiltonian.
        
			By sample here, we mean, generating a random Hamiltonian in the type of models defined in each class.
			
			If the calculation is done for T=0, the code will generate S(q) for the ground state
			If T!=0, then the code will generate the density matrix and generate the corresponding metric.
			If T!=0, then the code will generate the density matrix and generate the corresponding metric.
		
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
    """ RandomizerHamiltonianRandom provides a random deviation of non-zero terms in a given input Hamiltonian
		input parameters: widths for the deviations for couplings J and fields h.
    """
    def __init__(self, ham, delta, no_of_processes):
        """delta : tuple
			gives input for random variation in couplings and fields: (delta_J, delta_h)
        """
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
        exp_fac = params["exp_fac"]
        Lambdas = params["Lambdas"]
        Sq_int_in = params["Sq_int_in"]
        no_qpoints = params["no_qpoints"]
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
        zero_couplings=np.zeros((3,3))
        ranH_J = []
        for i in range(L):
            row=[]
            for j in range(L):
                if not np.allclose(J[i][j],zero_couplings): # only modify couplings that are already non-zero
                    tmp_J = delta[0] * (np.random.rand(3, 3) - 0.5)
                    tmp_J = tmp_J.conj().T + tmp_J # symmetrize the difference
                    row.append(J[i][j] + tmp_J)
                else:
                    row.append(zero_couplings)
            ranH_J.append(row)
			
        if (len(ranH_J)!=L):
            print("Wrong length of list ranH_J")
            exit(1)
        for i in range(L):
            if (len(ranH_J[i])!=L):
                print("Wrong length of list ranH_J[{}]".format(i))
                exit(1)
				
        ranH_h = delta[1] * (np.random.rand(3) - 0.5)

        H_plus_delta = create_hamiltonian_sparse(states, params_input={"L":L, "J": ranH_J,
                                                                       "h": h + ranH_h})
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
            Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.return_Sq2(L, state_rand, SX, SY, SZ, temp, no_ofqpoints=100, exp_fac=exp_fac, Lambdas=Lambdas)
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
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": 1/L/9 * sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand, "Sij_list": Sij_rand, "dist_ham": dist_ham, "rham": H_plus_delta, "ham_params": {"J": ranH_J, "h": h + ranH_h}, "Sq_int": sqrt(Sq_int_total), "Sq_int_list": Sq_int_rand}
        else:
            return {"Sij": 1/L/9 * sqrt(S_total), "Sq": 1/L/9 * sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand, "Sij_list": Sij_rand, "dist_ham": dist_ham, "Sq_int": sqrt(Sq_int_total), "Sq_int_list": Sq_int_rand}
            
		

class RandomizerHamiltonianRandomRandomDelta(Randomizer):
    """ randomize J and h, additionally multiplying each with a random amplitude between 0 and 1 for each realisation """
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
        """ see documentation for super().return_random_state() """
			
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
        """ see documentation for super().return_random_state() """
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
    """ class for randomising state vectors around a target state
        no further evolutions of this class have been implemented - for more advanced usage, see RandomizerStateRandomDelta, below.
    """
    def __init__(self, ham, delta, no_of_processes):
        self.ham = ham
        self.delta = delta
        self.no_of_processes = no_of_processes
        self.rand_ham = False
        self.rand_delta = False


    def return_random_state(self, params):
        """ see documentation for super().return_random_state() """
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

class RandomizerStateRandomDelta(RandomizerState):

    def __init__(self, ham, delta_max, no_of_processes, distribution_type):
        """
				Sample
				ham
				distribution_type: dict - important keys:
				
				'shape': allowed values are: 'uniformSquare', 'uniformCircle', 'normal'
				'rhoMethod': defining the algorithm for sampling random density matrices, allowed values are: 'squareRandom', 'projectEigenvalues'
				'rhoParams': further parameters for dm generator
				'scaleWithSize': flag indicating if scaling with Hilbert-space dimension should be considered
				
		"""
        self.ham = ham
        self.delta_max = delta_max
        self.no_of_processes = no_of_processes
        self.rand_ham = False
        self.rand_delta = True
        print  ("Setting up RandomizerStateRandomDelta")
        if not 'shape' in distribution_type:
            print("Please define the 'shape' of your distribution: 'uniformSquare', 'uniformCircle', or 'normal' ")
            exit(1)
        self.element_distribution = SelectElementDistribution(distribution_type['shape'])
        if ham.temp == 0.0:
            self.sampler = self._ZeroTemperatureSampler
        else:
            self.sampler = self._FiniteTemperatureSampler
            self.dm_sampler = SelectDensityMatrixSampler(self.element_distribution, {'method': distribution_type['rhoMethod'], 'params': distribution_type['rhoParams']})
        if 'scaleWithSize' in distribution_type:
            if distribution_type['scaleWithSize']:
                if ham.temp == 0.0:
                    self.delta_max *= np.pow(self.ham.dimension,-0.5)
                else:
                    self.delta_max *= np.pow(self.ham.dimension,-0.5)

    def return_random_state(self, params):
        """ see also documentation for super().return_random_state()
			params : dict
				description of reference system.
        """
        # super().return_random_state()
        delta = self.delta_max * np.random.rand(1)[0]
        return self.sampler(delta, params)
                    
    def _ZeroTemperatureSampler(self, delta, params):
        """ draw a random sample for a pure state """
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
        state_no = len(state_in)

        # draw random entries for each of the components of the state vector, and normalise it:
        state_rand = state_in + self.element_distribution.get_random_elements(delta, state_no)
        state_rand = state_rand / np.sqrt(np.sum(state_rand.conj() * state_rand))
        
        Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.return_Sq2(self.ham.L, state_rand, SX, SY, SZ, self.ham.temp, no_ofqpoints=no_qpoints,
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
        return {"Sij": 1/(9*self.ham.L) * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand,
                    "Sij_list": Sij_rand, "Sq_int": Sq_int_total, "Sq_int_list": Sq_int_rand}
                    
             
    def _FiniteTemperatureSampler(self, delta, params):
        """ draw a random density matrix for the finite temperature case """
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
        no_qpoints = params["no_qpoints"]	            # Here state_in must be dense
        # generation of actual random density matrix is contracted out to generator object:
        state_rand = self.dm_sampler.get_density_matrix(state_in, delta)
        Sij_rand, Sq_rand, Sq_int_rand = SijCalculator.returnSq2_dm_not_sparse(self.ham.L, state_rand, SX, SY, SZ,                 				no_ofqpoints=no_qpoints, exp_fac=exp_fac, Lambdas=Lambdas)
        S_total = 0
        Sq_total = 0
        for corr_i in corr:
            S_total += np.sum(np.abs(Sij_init[corr_i] - Sij_rand[corr_i])**2)
            Sq_total += np.sum(np.abs(Sq_init[corr_i] - Sq_rand[corr_i])**2)
        Sq_int_total = np.sum(np.abs(Sq_int_in - Sq_int_rand)**2)
        dist = 1/2 * np.abs(((state_in - state_rand).conj().T @ (state_in - state_rand)).trace())[0,0]
        energy = (H_in @ state_rand).trace()

        return {"Sij": 1/(9*self.ham.L) * sqrt(S_total), "Sq": sqrt(Sq_total), "dist": dist, "energy": energy, "Sq_list": Sq_rand,
                    "Sij_list": Sij_rand, "Sq_int": Sq_int_total, "Sq_int_list": Sq_int_rand}
 
